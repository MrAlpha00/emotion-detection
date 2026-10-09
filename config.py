# =============================================================================
# Emotion Detection Using Facial Expression - Configuration
# =============================================================================
# Centralized configuration for the entire application.
#
# The SAME codebase runs in two modes:
#
#   LOCAL DEVELOPMENT (no DATABASE_URL set)
#       SQLite  +  local filesystem storage (static/uploads, static/results)
#
#   PRODUCTION (DATABASE_URL set)
#       PostgreSQL (Supabase)  +  Supabase Storage
#
# Nothing in this file hardcodes a real credential. Every secret is read from
# the process environment (locally from the .env file, on Vercel from the
# project's Environment Variables).
# =============================================================================

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - python-dotenv is optional at runtime
    pass


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _env_bool(name, default=False):
    """Read a boolean-ish environment variable."""
    raw = os.getenv(name)
    if raw is None or raw == '':
        return default
    return raw.strip().lower() in ('true', '1', 'yes', 'on')


def _env_int(name, default):
    """Read an integer environment variable, falling back on any error."""
    raw = os.getenv(name)
    if raw is None or raw == '':
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def normalize_database_url(url):
    """
    Normalize a PostgreSQL connection string for SQLAlchemy + psycopg 3.

    - ``postgres://``            -> ``postgresql://``  (legacy Supabase scheme)
    - ``postgresql://``          -> ``postgresql+psycopg://``
    - ``postgresql+psycopg2://`` -> ``postgresql+psycopg://``  (obsolete driver)
    - ``postgresql://...?sslmode=require`` keeps its query parameters.

    Already-correct URLs (``postgresql+psycopg://``) are returned untouched.
    """
    if not url:
        return url

    url = url.strip()

    # Warn about the single most common copy/paste mistake: an unescaped '@'
    # inside the password. SQLAlchemy splits on the LAST '@', so the real host
    # ends up in the password field and the host looks like garbage. The
    # password cannot be repaired here - the correct value must be
    # percent-encoded in the environment variable - but the failure is otherwise
    # a confusing DNS error.
    scheme, _, rest = url.partition('://')
    if '@' in rest:
        userinfo, _, hostpart = rest.rpartition('@')
        if '@' in userinfo:
            import sys

            print(
                'WARNING: DATABASE_URL contains an unescaped "@" in the password '
                'section. Percent-encode it as %%40 (and any other reserved '
                'characters) or the connection will fail with a misleading DNS '
                'error. The value itself is never printed.',
                file=sys.stderr,
            )

    # Legacy Supabase / Heroku style scheme
    if url.startswith('postgres://'):
        url = 'postgresql://' + url[len('postgres://'):]

    # Force the modern psycopg 3 driver unless an explicit driver is given
    if url.startswith('postgresql://'):
        url = 'postgresql+psycopg://' + url[len('postgresql://'):]
    elif url.startswith('postgresql+psycopg2://'):
        url = 'postgresql+psycopg://' + url[len('postgresql+psycopg2://'):]

    return url


def sqlite_uri_from_url(url, fallback_path):
    """
    Build an absolute SQLAlchemy SQLite URI for a ``sqlite:///`` URL.

    Two traps are handled here:

    1. Flask-SQLAlchemy resolves a *relative* SQLite path against
       ``app.instance_path``. So ``sqlite:///database/app.db`` silently opens
       ``./instance/database/app.db`` instead of ``./database/app.db``, and
       fails outright when that directory does not exist. The path is anchored
       to BASE_DIR so a developer-supplied relative URL points where it reads.
    2. Windows paths need forward slashes. ``sqlite:///C:\\Users\\...`` is not a
       valid SQLite URI; ``sqlite:///C:/Users/...`` is.

    ``fallback_path`` is used when the URL carries no path at all.
    """
    from pathlib import Path

    raw = ''
    if url:
        _, sep, remainder = url.partition(':///')
        if sep:
            raw = remainder.split('?', 1)[0]

    path = Path(raw) if raw else Path(fallback_path)
    if not path.is_absolute():
        path = Path(BASE_DIR) / path

    return 'sqlite:///' + path.as_posix()


def redact_database_url(url):
    """
    Return a log-safe representation of a database URL.

    The password is replaced with ``***`` so connection information can be
    logged without leaking credentials.
    """
    if not url:
        return ''
    try:
        scheme, _, rest = url.partition('://')
        if '@' in rest:
            _, _, host = rest.rpartition('@')
            return f'{scheme}://***:***@{host}'
        return f'{scheme}://***'
    except Exception:
        return '***'


# Base directory of the project
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


# -----------------------------------------------------------------------------
# Emotion class labels (see the long comment inside Config for the rationale)
# -----------------------------------------------------------------------------
DEFAULT_EMOTION_LABELS = [
    'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise',
]


def _emotion_labels_from_env():
    """
    Read an optional JSON array of class labels from the environment.

    Lets an operator who trained their own checkpoint state the real class
    order explicitly instead of inheriting this project's documented default.
    Anything invalid falls back to the default rather than crashing the app.
    """
    import json
    import sys

    raw = (os.getenv('EMOTION_LABELS_JSON') or '').strip()
    if not raw:
        return list(DEFAULT_EMOTION_LABELS)

    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        print(
            'WARNING: EMOTION_LABELS_JSON is not valid JSON; falling back to the '
            'documented default order.',
            file=sys.stderr,
        )
        return list(DEFAULT_EMOTION_LABELS)

    if not isinstance(parsed, list) or not parsed:
        print(
            'WARNING: EMOTION_LABELS_JSON must be a non-empty JSON array; falling back '
            'to the documented default order.',
            file=sys.stderr,
        )
        return list(DEFAULT_EMOTION_LABELS)

    labels = [str(item).strip() for item in parsed]
    if any(not label for label in labels):
        print(
            'WARNING: EMOTION_LABELS_JSON contained an empty label; falling back to the '
            'documented default order.',
            file=sys.stderr,
        )
        return list(DEFAULT_EMOTION_LABELS)

    return labels


class Config:
    """Main application configuration."""

    BASE_DIR = BASE_DIR

    # -------------------------------------------------------------------------
    # Flask Settings
    # -------------------------------------------------------------------------
    # A development fallback keeps `python app.py` working out of the box. It is
    # NOT used in production: ProductionConfig below raises when SECRET_KEY is
    # missing, because a known default signing key would let anyone forge a
    # session cookie and become an admin.
    SECRET_KEY = os.getenv('SECRET_KEY') or 'emotion-detection-secret-key-change-in-production'
    DEBUG = _env_bool('FLASK_DEBUG', False)

    # Server-side session cookie hardening. On plain HTTP (local dev) the
    # Secure flag would prevent the cookie from being sent, so it is only
    # enabled when the app is actually running behind HTTPS.
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    SESSION_COOKIE_SECURE = _env_bool('SESSION_COOKIE_SECURE', _env_bool('VERCEL', False))
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = 'Lax'
    REMEMBER_COOKIE_SECURE = SESSION_COOKIE_SECURE
    PERMANENT_SESSION_LIFETIME = 60 * 60 * 24 * 7  # 7 days

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------
    DATABASE_DIR = os.path.join(BASE_DIR, 'database')
    DATABASE_PATH = os.path.join(DATABASE_DIR, 'emotion_app.db')
    DATABASE_FILENAME = 'emotion_app.db'

    # Raw value as supplied by the environment (may be empty).
    DATABASE_URL = os.getenv('DATABASE_URL') or ''
    DATABASE_URL_NORMALIZED = normalize_database_url(DATABASE_URL)

    # The dialect is derived from the URI scheme, NOT from whether DATABASE_URL
    # happens to be non-empty. Setting DATABASE_URL to a sqlite:/// URL is a
    # legitimate way to point at a local file, and mislabelling it
    # 'postgresql' previously made the app skip creating the database directory
    # and then fail to open the file.
    _URL_SCHEME = DATABASE_URL_NORMALIZED.split(':', 1)[0].lower()
    _USES_SQLITE_URL = _URL_SCHEME in ('sqlite', 'sqlite+pysqlite')

    if DATABASE_URL.strip() and not _USES_SQLITE_URL:
        # ---------------------------------------------------------------------
        # PRODUCTION: PostgreSQL
        # ---------------------------------------------------------------------
        SQLALCHEMY_DATABASE_URI = DATABASE_URL_NORMALIZED
        SQLALCHEMY_ENGINE_OPTIONS = {
            'pool_pre_ping': True,
            'pool_recycle': 280,
            'pool_size': 5,
            'max_overflow': 10,
            'pool_timeout': 30,
        }
        DB_DIALECT = 'postgresql'
        IS_PRODUCTION_DB = True
    else:
        # ---------------------------------------------------------------------
        # LOCAL DEVELOPMENT: SQLite (never used in production)
        # ---------------------------------------------------------------------
        # sqlite_uri_from_url() anchors the path to BASE_DIR and converts
        # backslashes to forward slashes, so the file really lives at
        # ./database/emotion_app.db on every platform.
        SQLALCHEMY_DATABASE_URI = sqlite_uri_from_url(
            DATABASE_URL, DATABASE_PATH
        )
        SQLALCHEMY_ENGINE_OPTIONS = {}
        DB_DIALECT = 'sqlite'
        IS_PRODUCTION_DB = False

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # True when the SQLite file above is the active backend. Used by the app
    # factory to decide whether creating local directories is appropriate.
    USE_SQLITE = not IS_PRODUCTION_DB

    # -------------------------------------------------------------------------
    # Supabase
    # -------------------------------------------------------------------------
    # SUPABASE_URL is the project REST endpoint, e.g.
    #   https://<project-ref>.supabase.co
    SUPABASE_URL = (os.getenv('SUPABASE_URL') or '').strip() or None

    # Server-side only. Resolution order:
    #   1. SUPABASE_SECRET_KEY   (current naming, server-only secret)
    #   2. SUPABASE_KEY          (backwards compatible fallback)
    # When the first candidate is a publishable key the second is still
    # considered, because a publishable/anon key cannot perform privileged
    # Storage operations and would otherwise disable the backend even though a
    # usable secret is present in the other variable.
    SUPABASE_SECRET_KEY = (os.getenv('SUPABASE_SECRET_KEY') or '').strip() or None
    SUPABASE_KEY_LEGACY = (os.getenv('SUPABASE_KEY') or '').strip() or None

    _candidates = [k for k in (SUPABASE_SECRET_KEY, SUPABASE_KEY_LEGACY) if k]

    def _is_publishable(value):
        return bool(value) and value.startswith('sb_publishable_')

    _resolved_supabase_key = None
    for _candidate in _candidates:
        if not _is_publishable(_candidate):
            _resolved_supabase_key = _candidate
            break
    if _resolved_supabase_key is None and _candidates:
        # Only publishable keys are present.
        _resolved_supabase_key = _candidates[0]

    if _resolved_supabase_key and _is_publishable(_resolved_supabase_key):
        # A publishable key cannot perform privileged Storage operations.
        SUPABASE_KEY_IS_PUBLISHABLE = True
    else:
        SUPABASE_KEY_IS_PUBLISHABLE = False

    SUPABASE_KEY = _resolved_supabase_key

    @classmethod
    def supabase_credentials_ready(cls):
        """True when both URL and a non-publishable key are available."""
        return bool(cls.SUPABASE_URL and cls.SUPABASE_KEY and not cls.SUPABASE_KEY_IS_PUBLISHABLE)

    # -------------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------------
    # 'local'    -> static/uploads + static/results on the local filesystem
    # 'supabase' -> private Supabase Storage bucket
    STORAGE_BACKEND = (os.getenv('STORAGE_BACKEND') or 'local').strip().lower()
    STORAGE_BUCKET = (os.getenv('STORAGE_BUCKET') or 'emotion-images').strip()
    SIGNED_URL_TTL_SECONDS = _env_int('SIGNED_URL_TTL_SECONDS', 300)

    # -------------------------------------------------------------------------
    # File uploads
    # -------------------------------------------------------------------------
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
    MAX_CONTENT_LENGTH = _env_int('UPLOAD_LIMIT', 16 * 1024 * 1024)  # 16 MB
    UPLOAD_LIMIT = MAX_CONTENT_LENGTH
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    ALLOWED_MIME_TYPES = {'image/jpeg', 'image/jpg', 'image/png'}

    # -------------------------------------------------------------------------
    # Pagination
    # -------------------------------------------------------------------------
    DEFAULT_PAGE_SIZE = _env_int('PAGE_SIZE', 25)
    PAGE_SIZE_OPTIONS = [25, 50, 100]
    MAX_PAGE_SIZE = 200

    # -------------------------------------------------------------------------
    # Emotion detection model
    # -------------------------------------------------------------------------
    MODEL_PATH = os.getenv('MODEL_PATH') or os.path.join(BASE_DIR, 'model', 'finalfacialemotionmodel.keras')

    # Expected model input. 48x48 grayscale is the FER-2013 standard.
    MODEL_INPUT_SIZE = (48, 48)
    MODEL_COLOR_MODE = 'grayscale'
    MODEL_CHANNELS_LAST = True           # (None, H, W, C) layout
    MODEL_NORMALIZATION_FACTOR = 255.0   # pixel / 255 -> [0, 1]

    # -------------------------------------------------------------------------
    # Emotion class labels
    # -------------------------------------------------------------------------
    # ORDER MATTERS - getting this wrong silently mislabels every prediction.
    #
    # Evidence for this order:
    #   * model/finalfacialemotionmodel.keras is the checkpoint documented at
    #     https://huggingface.co/lokeshkumar79/facial-emotion-recognition,
    #     whose output order is
    #     [Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise].
    #   * The architecture is the canonical Keras CNN FER-2013 example
    #     (Conv2D/BN/MaxPool/Dropout/Dense, 48x48x1 input, 7-way softmax).
    #
    # LIMITATION: the saved .keras archive does NOT embed the class names (its
    # config.json carries no label metadata and model.weights.h5 has no label
    # attributes), so the order cannot be proven from the file itself. It is
    # therefore overridable through the EMOTION_LABELS_JSON environment
    # variable for anyone who trained this checkpoint themselves.
    #
    # The predictor rejects the model when its output dimension disagrees with
    # the length of this list, so a wrong *number* of classes can never be
    # silently mapped.
    EMOTION_LABELS = _emotion_labels_from_env()

    NUM_EMOTION_CLASSES = len(EMOTION_LABELS)

    # -------------------------------------------------------------------------
    # Live detection / abuse protection
    # -------------------------------------------------------------------------
    # Minimum seconds between two processed frames from the same session.
    LIVE_DETECTION_INTERVAL = _env_int('LIVE_DETECTION_INTERVAL', 2)
    # Hard ceiling on a single live session, in seconds.
    MAX_LIVE_SESSION_SECONDS = _env_int('MAX_LIVE_SESSION_SECONDS', 60 * 60)
    # Simple per-session/per-user token bucket used by the live frame endpoint.
    RATE_LIMIT_WINDOW_SECONDS = _env_int('RATE_LIMIT_WINDOW_SECONDS', 60)
    RATE_LIMIT_MAX_REQUESTS = _env_int('RATE_LIMIT_MAX_REQUESTS', 40)

    # -------------------------------------------------------------------------
    # Timestamp storage / display timezones
    # -------------------------------------------------------------------------
    # STORAGE is always UTC: every column is a naive db.DateTime written from a
    # timezone-aware datetime.now(timezone.utc) value (see models/*.py). Naive
    # values read back from the database are therefore interpreted as UTC.
    #
    # DISPLAY converts those UTC values through this IANA zone before anything
    # is rendered or exported. It is a proper zone conversion (zoneinfo), never
    # a hardcoded "+5:30" offset, so daylight-saving rules of the zone apply.
    DISPLAY_TIMEZONE = (os.getenv('DISPLAY_TIMEZONE') or '').strip() or 'Asia/Kolkata'

    # Label appended to exported column headers so the sheet states which zone
    # its timestamps are in (e.g. "Timestamp (IST)").
    DISPLAY_TIMEZONE_LABEL = (
        (os.getenv('DISPLAY_TIMEZONE_LABEL') or '').strip() or 'IST'
    )

    # -------------------------------------------------------------------------
    # Timestamp display formats
    # -------------------------------------------------------------------------
    TIMESTAMP_DISPLAY_FORMAT = '%d %B %Y, %I:%M:%S %p'
    DATE_DISPLAY_FORMAT = '%d %b %Y'
    TIME_DISPLAY_FORMAT = '%I:%M:%S %p'

    # -------------------------------------------------------------------------
    # Prediction diagnostics
    # -------------------------------------------------------------------------
    # When enabled, every inference logs the full per-face probability vector
    # (one entry per emotion class, in %) to the "diagnostics.prediction"
    # logger. This exists purely to evaluate model quality from the logs; it
    # never influences the predicted class or the stored result.
    PREDICT_LOG_PROBABILITIES = _env_bool('PREDICT_LOG_PROBABILITIES', True)

    # -------------------------------------------------------------------------
    # Face detection (OpenCV Haar Cascade)
    # -------------------------------------------------------------------------
    FACE_DETECTION_SCALE_FACTOR = 1.3
    FACE_DETECTION_MIN_NEIGHBORS = 5
    FACE_DETECTION_MIN_SIZE = (30, 30)

    # -------------------------------------------------------------------------
    # Roles
    # -------------------------------------------------------------------------
    ROLE_USER = 'user'
    ROLE_ADMIN = 'admin'
    ALLOWED_ROLES = (ROLE_USER, ROLE_ADMIN)

    # -------------------------------------------------------------------------
    # Startup behaviour
    # -------------------------------------------------------------------------
    # When True the app creates missing tables on boot. Convenient for a local
    # SQLite file, and safe there because create_all() never drops or alters an
    # existing table.
    #
    # It defaults to OFF whenever DATABASE_URL points at PostgreSQL, so a
    # developer who has production credentials in .env cannot accidentally let
    # create_all() build a schema behind Alembic's back. PostgreSQL schema
    # changes go through `flask db upgrade`.
    AUTO_CREATE_TABLES = _env_bool('AUTO_CREATE_TABLES', not IS_PRODUCTION_DB)

    # -------------------------------------------------------------------------
    # CSRF
    # -------------------------------------------------------------------------
    CSRF_ENABLED = _env_bool('CSRF_ENABLED', True)


class DevelopmentConfig(Config):
    """Local development defaults (SQLite + local storage)."""

    DEBUG = True
    STORAGE_BACKEND = 'local'
    AUTO_CREATE_TABLES = True


class ProductionConfig(Config):
    """Production defaults (PostgreSQL + Supabase Storage)."""

    DEBUG = False
    STORAGE_BACKEND = 'supabase'
    SESSION_COOKIE_SECURE = True
    AUTO_CREATE_TABLES = False

    def __init__(self):
        super().__init__()

        # Fail loudly and early rather than signing cookies with a key that is
        # published in this repository.
        if self.SECRET_KEY == Config.SECRET_KEY and (os.getenv('SECRET_KEY') or '').strip() == '':
            raise RuntimeError(
                'SECRET_KEY is not set. Generate one with '
                '`python -c "import secrets; print(secrets.token_hex(32))"` and set it '
                'as an environment variable (on Vercel: Project Settings > Environment '
                'Variables). Refusing to start with the development fallback key.'
            )

        if not self.DATABASE_URL.strip():
            raise RuntimeError(
                'DATABASE_URL is not set. Production requires PostgreSQL; it is never '
                'silently downgraded to a local SQLite file.'
            )

        if not self.supabase_credentials_ready():
            raise RuntimeError(
                'Supabase storage is not usable: SUPABASE_URL and a server-side '
                'SUPABASE_SECRET_KEY (sb_secret_... or legacy service_role JWT) are both '
                'required. The publishable key cannot manage a private bucket.'
            )


def get_config():
    """
    Return the configuration class appropriate for the current environment.

    ``FLASK_ENV=production`` (which Vercel sets automatically) selects
    ProductionConfig, otherwise Config is used. ProductionConfig still derives
    the database URI from ``DATABASE_URL``, so a misconfigured deployment
    surfaces as a clear error rather than a silent fallback to SQLite.
    """
    env = (os.getenv('FLASK_ENV') or '').strip().lower()
    if env == 'production' or _env_bool('VERCEL', False):
        return ProductionConfig
    return Config


def validate_production_config():
    """
    Instantiate the selected config so a misconfigured production deployment
    fails immediately with an actionable message.

    This is called by ``python -c "import config; config.validate_production_config()"``
    in the deployment checklist and by the application factory before any
    database work is attempted.
    """
    return get_config()()
