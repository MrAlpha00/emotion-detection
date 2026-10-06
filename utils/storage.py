# =============================================================================
# Storage Abstraction
# =============================================================================
# One interface, two backends:
#
#   STORAGE_BACKEND=local     -> static/uploads and static/results on disk
#   STORAGE_BACKEND=supabase  -> private Supabase Storage bucket
#
# Object keys are always generated server-side from a UUID plus the owner id;
# client supplied filenames are never used as paths.
#
# Images are handed back to the browser in one of two safe ways:
#   * LOCAL mode    -> a signed, ownership-checked route streams the file
#   * SUPABASE mode -> a short-lived signed URL, or a streamed download
# =============================================================================

import io
import logging
import os
import re
import uuid
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)

# Only ever allow these suffixes to be written by this module.
_SAFE_SUFFIXES = {'.jpg', '.jpeg', '.png'}


def safe_extension(filename=None, fallback='jpg'):
    """
    Derive a safe lowercase extension from a client supplied filename.

    Returns ``None`` when the extension is not an allowed image type, which
    callers must treat as a rejected upload.
    """
    if not filename:
        return '.' + fallback
    suffix = Path(str(filename)).suffix.lower()
    if suffix in _SAFE_SUFFIXES:
        return suffix
    return None


def build_object_key(user_id, kind, suffix='.jpg'):
    """
    Build a collision-free storage key.

        users/<user_id>/uploads/<uuid>.jpg
        users/<user_id>/results/<uuid>.jpg

    ``user_id`` is coerced to a positive integer so a hostile value can never
    escape into the path.
    """
    kind = 'uploads' if kind == 'uploads' else 'results'
    try:
        safe_user_id = int(user_id)
    except (TypeError, ValueError):
        raise ValueError('user_id must be an integer')

    name = f'{uuid.uuid4().hex}{suffix}'
    return f'users/{safe_user_id}/{kind}/{name}'


def is_object_key_owned_by(key, user_id):
    """
    Ownership check for a storage key.

    Returns True only when the key is shaped like ``users/<user_id>/...`` and
    the embedded user id matches. This is what stops one user from reading or
    deleting another user's image by editing a URL.
    """
    if not key or user_id is None:
        return False
    match = re.match(r'^users/(\d+)/(uploads|results)/[A-Za-z0-9._-]+$', str(key))
    if not match:
        return False
    return int(match.group(1)) == int(user_id)


def is_valid_object_key(key):
    """True when the key looks like something this application created."""
    if not key:
        return False
    return bool(re.match(r'^users/\d+/(uploads|results)/[A-Za-z0-9._-]+$', str(key)))


class StorageError(RuntimeError):
    """Raised when a storage operation cannot be completed."""


# Anything shaped like a credential is scrubbed before an exception is logged.
# The Storage API carries its key in a header rather than the URL, so this is
# defence in depth against a lower-level client error that embeds a full URL.
_SECRET_PATTERNS = (
    re.compile(r'sb_secret_[A-Za-z0-9]+'),
    re.compile(r'sb_publishable_[A-Za-z0-9]+'),
    re.compile(r'eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}'),
)


def describe_storage_error(exc):
    """
    Render a Storage API failure as a short, credential-free log detail.

    storage3 raises ``StorageApiError``, whose ``str()`` is a dict-like repr
    such as ``{'statusCode': 404, 'error': 'Bucket not found', ...}``. The
    structured fields are read instead, so a deployment log line reads like a
    report - ``StorageApiError, status=404, code=Bucket not found,
    message=Bucket not found`` - rather than a Python repr. The result is
    length-capped and scrubbed of anything key-shaped. Bucket names and object
    keys are not secrets; the caller logs those.
    """
    parts = [type(exc).__name__]
    for attr in ('status', 'code', 'message'):
        value = getattr(exc, attr, None)
        if value not in (None, ''):
            parts.append(f'{attr}={value}')
    if len(parts) == 1:
        # Not a storage3 error (network failure, timeout, ...).
        parts.append(str(exc))
    text = ', '.join(parts)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub('[redacted]', text)
    return text[:400]


class BaseStorage:
    """Common interface for every storage backend."""

    backend = 'base'

    def save(self, image_bytes, user_id, kind='uploads', suffix='.jpg'):
        """Persist bytes and return the storage key."""
        raise NotImplementedError

    def load(self, key):
        """Return ``(bytes, content_type)`` for a key, or ``None``."""
        raise NotImplementedError

    def delete(self, key):
        """Delete a key. Missing objects are treated as success."""
        raise NotImplementedError

    def url_for(self, key, ttl=None):
        """Return a directly usable URL, or ``None`` if streaming is required."""
        return None

    def describe(self):
        """Short, credential-free description used for startup logging."""
        return self.backend


class LocalStorage(BaseStorage):
    """
    Filesystem storage rooted outside of ``static/`` so that images are never
    exposed by the static file handler.

    Layout::

        <BASE_DIR>/instance/uploads/users/<user_id>/uploads/<uuid>.jpg
        <BASE_DIR>/instance/uploads/users/<user_id>/results/<uuid>.jpg
    """

    backend = 'local'

    def __init__(self, root=None):
        # BASE_DIR is a plain string in config, so it must be wrapped in Path
        # before the '/' operator can be used.
        default_root = Path(Config.BASE_DIR) / 'instance' / 'uploads'
        self.root = Path(root or default_root).resolve()

    def _absolute(self, key):
        """Resolve a key to an absolute path, refusing any traversal attempt."""
        if not is_valid_object_key(key):
            raise StorageError('Invalid storage key')
        candidate = (self.root / str(key)).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise StorageError('Storage key escapes the storage root')
        return candidate

    def save(self, image_bytes, user_id, kind='uploads', suffix='.jpg'):
        key = build_object_key(user_id, kind, suffix)
        path = self._absolute(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image_bytes)
        return key

    def load(self, key):
        try:
            path = self._absolute(key)
        except StorageError:
            return None
        if not path.is_file():
            return None
        content_type = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
        return path.read_bytes(), content_type

    def delete(self, key):
        try:
            path = self._absolute(key)
        except StorageError:
            return
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning('Could not delete local object %s: %s', key, exc)

    def url_for(self, key, ttl=None):
        # Local objects are streamed through an authenticated route instead of
        # being served as public static files.
        return None

    def describe(self):
        return 'LOCAL'


class SupabaseStorage(BaseStorage):
    """Private Supabase Storage backend using the server-side secret key."""

    backend = 'supabase'

    def __init__(self, bucket=None, ttl=None):
        if not Config.supabase_credentials_ready():
            raise StorageError(
                'Supabase storage requires SUPABASE_URL and a server-side '
                'SUPABASE_SECRET_KEY. The publishable/anon key cannot be used.'
            )
        try:
            from supabase import create_client
        except Exception as exc:  # pragma: no cover - dependency missing
            raise StorageError(f'supabase package is not installed: {exc}')

        self.bucket = bucket or Config.STORAGE_BUCKET
        self.ttl = ttl or Config.SIGNED_URL_TTL_SECONDS
        # The service/secret key is used here and only here, server-side.
        self._client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

    def save(self, image_bytes, user_id, kind='uploads', suffix='.jpg'):
        if not image_bytes:
            raise StorageError('Refusing to store an empty object')
        key = build_object_key(user_id, kind, suffix)
        content_type = 'image/png' if suffix.lower() == '.png' else 'image/jpeg'
        try:
            self._client.storage.from_(self.bucket).upload(
                key,
                image_bytes,
                file_options={'content-type': content_type, 'upsert': 'false'},
            )
        except Exception as exc:
            raise StorageError(
                f'Supabase upload failed (bucket={self.bucket!r}, key={key!r}): '
                f'{describe_storage_error(exc)}'
            ) from exc
        return key

    def load(self, key):
        if not is_valid_object_key(key):
            return None
        try:
            data = self._client.storage.from_(self.bucket).download(key)
        except Exception as exc:
            logger.warning(
                'Supabase download failed (bucket=%r, key=%s): %s',
                self.bucket, key, describe_storage_error(exc),
            )
            return None
        content_type = 'image/png' if str(key).lower().endswith('.png') else 'image/jpeg'
        if isinstance(data, (bytes, bytearray)):
            return bytes(data), content_type
        if isinstance(data, dict) and 'content' in data:  # pragma: no cover
            return bytes(data['content']), content_type
        # Unexpected payload shape: report "missing" rather than returning a
        # half-empty tuple that a caller would try to stream.
        return None

    def delete(self, key):
        if not is_valid_object_key(key):
            return
        try:
            self._client.storage.from_(self.bucket).remove([key])
        except Exception as exc:
            logger.warning(
                'Supabase delete failed (bucket=%r, key=%s): %s',
                self.bucket, key, describe_storage_error(exc),
            )

    def url_for(self, key, ttl=None):
        if not is_valid_object_key(key):
            return None
        try:
            signed = self._client.storage.from_(self.bucket).create_signed_url(
                key, ttl or self.ttl
            )
        except Exception as exc:
            logger.warning(
                'Could not create a signed URL (bucket=%r, key=%s): %s',
                self.bucket, key, describe_storage_error(exc),
            )
            return None
        if isinstance(signed, dict):
            # supabase-py v2 returns {'signedURL': '/object/sign/...', 'signedUrl': ...}
            return signed.get('signedURL') or signed.get('signedUrl') or signed.get('url')
        return signed

    def describe(self):
        return 'SUPABASE'


def _build_storage():
    """
    Instantiate the configured backend.

    An unrecognised STORAGE_BACKEND is a configuration error, not something to
    paper over by silently using local disk in production.
    """
    backend = (Config.STORAGE_BACKEND or 'local').strip().lower()

    if backend not in ('local', 'supabase'):
        logger.error(
            'Unknown STORAGE_BACKEND %r. Expected "local" or "supabase".', backend
        )
        return UnavailableStorage(backend, f'Unknown STORAGE_BACKEND {backend!r}')

    if backend == 'supabase':
        try:
            return SupabaseStorage()
        except StorageError as exc:
            # Never silently pretend images are being persisted. Refuse loudly
            # but keep the app importable so /health and the admin UI still work.
            logger.error('Supabase storage unavailable: %s', exc)
            return UnavailableStorage('supabase', str(exc))

    return LocalStorage()


class UnavailableStorage(BaseStorage):
    """
    Placeholder used when the configured backend cannot be initialised.

    Every mutating call raises, so a misconfigured production deployment fails
    loudly on the first upload instead of silently losing data.
    """

    def __init__(self, backend, reason):
        self.backend = backend
        self.reason = reason

    def save(self, *args, **kwargs):
        raise StorageError(f'Storage backend unavailable: {self.reason}')

    def load(self, key):
        return None

    def delete(self, key):
        return

    def url_for(self, key, ttl=None):
        return None

    def describe(self):
        return self.backend.upper()


# Module-level singleton, created lazily so importing this module never touches
# the network or the filesystem during app import.
_storage = None


def get_storage():
    """Return the shared storage instance, constructing it on first use."""
    global _storage
    if _storage is None:
        _storage = _build_storage()
    return _storage


def reset_storage():
    """Drop the cached instance. Used by tests."""
    global _storage
    _storage = None


def describe_storage():
    """Credential-free storage description for startup logs."""
    try:
        return get_storage().describe()
    except Exception:
        return 'UNKNOWN'


def encode_image(image):
    """
    Encode an OpenCV BGR image to JPEG bytes.

    Returns ``None`` when encoding fails so callers never persist corrupt data.
    """
    import cv2

    if image is None or getattr(image, 'size', 0) == 0:
        return None
    ok, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return None
    return buffer.tobytes()


def encode_png(image):
    """Encode an OpenCV image to PNG bytes."""
    import cv2

    if image is None or getattr(image, 'size', 0) == 0:
        return None
    ok, buffer = cv2.imencode('.png', image)
    if not ok:
        return None
    return buffer.tobytes()


def bytes_io(data):
    """Wrap bytes in a BytesIO (helper for streaming responses)."""
    return io.BytesIO(data)


def local_fallback_path(relative_static_path):
    """
    Resolve a legacy ``static/...`` relative path for the pre-migration
    detection rows that still reference the original upload folders.

    Returns ``None`` when the file does not exist. This keeps historical rows
    viewable after the storage move without re-uploading anything.
    """
    if not relative_static_path:
        return None
    normalized = str(relative_static_path).replace('\\', '/')
    if normalized.startswith('static/'):
        normalized = normalized[len('static/'):]
    candidate = (Path(Config.BASE_DIR) / 'static' / normalized).resolve()
    try:
        candidate.relative_to(Path(Config.BASE_DIR) / 'static')
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


__all__ = [
    'BaseStorage',
    'LocalStorage',
    'StorageError',
    'SupabaseStorage',
    'UnavailableStorage',
    'build_object_key',
    'bytes_io',
    'describe_storage_error',
    'describe_storage',
    'encode_image',
    'encode_png',
    'get_storage',
    'is_object_key_owned_by',
    'is_valid_object_key',
    'local_fallback_path',
    'reset_storage',
    'safe_extension',
]
