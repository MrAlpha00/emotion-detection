# =============================================================================
# Detection Routes
# =============================================================================
# Image upload, camera capture, face detection, emotion inference, result
# display, live camera detection and result management.
#
# Design points relevant to production:
#   * Images are persisted through utils.storage, so the same code writes to the
#     local filesystem in development and to Supabase Storage in production.
#   * Storage keys are generated server-side from a UUID plus the owner id. The
#     client's filename is never used as a path.
#   * Images are only ever displayed through an authenticated, ownership-checked
#     route (/detection/image/<id>), never as a public static URL.
#   * Live detection is browser-driven: getUserMedia() captures a frame and
#     POSTs it here. The server never tries to open a webcam.
#   * Every frame is rate limited and only periodic detection *events* are
#     stored - never the raw video stream.
# =============================================================================

import base64
import binascii
import hmac
import logging
import secrets
import time
from datetime import datetime, timezone

import cv2
import numpy as np
from flask import (
    Blueprint,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
    flash,
)
from flask_login import current_user

from config import Config
from models.detection import Detection
from models.live_session import LiveSession
from models.user_activity import ActivityType
from utils.database import db
from utils.emotion_predictor import ModelUnavailableError, emotion_predictor
from utils.face_detector import (
    crop_face,
    detect_faces,
    draw_emotion_boxes,
    get_largest_face,
)
from utils.rate_limit import live_frame_limiter
from utils.security import active_user_required, log_activity, owns_object
from utils.storage import (
    StorageError,
    encode_image,
    get_storage,
    is_object_key_owned_by,
    is_valid_object_key,
    local_fallback_path,
    safe_extension,
)

logger = logging.getLogger(__name__)

detection_bp = Blueprint('detection', __name__)

# Hard ceiling on a decoded image, independent of the request body limit.
MAX_IMAGE_PIXELS = 12_000_000  # ~12 MP


def hmac_compare(a, b):
    """Constant-time string comparison for capability tokens."""
    return hmac.compare_digest((a or '').encode('utf-8'), (b or '').encode('utf-8'))


# =============================================================================
# Helpers
# =============================================================================
def allowed_file(filename):
    """True when the extension is one of the allowed image types."""
    suffix = safe_extension(filename)
    return suffix is not None


def decode_image_payload(payload):
    """
    Decode a base64 data URL into an OpenCV BGR image.

    Returns ``(image, error_message)``. ``image`` is None on failure.
    """
    if not payload or not isinstance(payload, str):
        return None, 'No image data received.'

    # Accept both a bare base64 string and a full data URL.
    if ',' in payload:
        payload = payload.split(',', 1)[1]

    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return None, 'The image data could not be decoded.'

    if not raw:
        return None, 'The image data was empty.'

    buffer = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if image is None:
        return None, 'The uploaded file is not a valid image or is corrupt.'

    if image.size == 0:
        return None, 'The uploaded file is not a valid image or is corrupt.'

    height, width = image.shape[:2]
    if height * width > MAX_IMAGE_PIXELS:
        return None, 'The image resolution is too large. Please use a smaller image.'

    return image, None


def _store_detection_images(original_image, annotated_image, user_id):
    """
    Persist the original and annotated images.

    Returns ``(original_key, processed_key)`` or raises StorageError. Images are
    encoded in memory, so no temporary file is ever written to disk - which
    matters because /tmp is not persistent on a serverless host.
    """
    storage = get_storage()

    original_bytes = encode_image(original_image)
    processed_bytes = encode_image(annotated_image)

    if original_bytes is None or processed_bytes is None:
        raise StorageError('Could not encode the images for storage.')

    original_key = storage.save(original_bytes, user_id, kind='uploads', suffix='.jpg')
    processed_key = storage.save(processed_bytes, user_id, kind='results', suffix='.jpg')
    return original_key, processed_key


def _model_unavailable_response():
    """503 payload used whenever the model is missing or invalid."""
    return jsonify({
        'success': False,
        'error': 'Emotion model unavailable. ' + emotion_predictor.get_status_message(),
        'model_available': False,
    }), 503


# =============================================================================
# Detection page
# =============================================================================
@detection_bp.route('/detect')
@active_user_required
def detect_page():
    """Upload / capture page."""
    if not emotion_predictor.ensure_loaded():
        logger.warning('Detection page opened without a usable model')
    return render_template('face_detection.html')


# =============================================================================
# Shared image analysis
# =============================================================================
def _analyse_and_store(image, detection_type):
    """
    Run face detection + emotion inference, persist images, return the result.

    Raises ModelUnavailableError or StorageError on failure. Never returns a
    fabricated prediction.
    """
    t0 = time.perf_counter()
    t_face = 0.0
    t_inf = 0.0
    t_store = 0.0

    t1 = time.perf_counter()
    faces = detect_faces(image)
    t_face = (time.perf_counter() - t1) * 1000.0
    if not faces:
        return None, 'No face was detected in the image. Please try a clearer photo.'

    primary_face = get_largest_face(faces)
    face_crop = crop_face(image, primary_face)
    if face_crop is None or face_crop.size == 0:
        return None, 'Failed to crop the detected face. Please try another image.'

    t1 = time.perf_counter()
    # face_context is diagnostic only: it labels the PREDICT_PROBS log line so
    # the raw probability vector can be traced back to a specific face.
    prediction = emotion_predictor.predict(
        face_crop,
        face_context=f'{detection_type}: largest of {len(faces)} face(s)',
    )  # raises if unavailable
    t_inf = (time.perf_counter() - t1) * 1000.0

    processing_time = round(time.perf_counter() - t0, 3)

    annotated = draw_emotion_boxes(
        image, [primary_face], [prediction['emotion']], [prediction['confidence']]
    )

    t1 = time.perf_counter()
    upload_key, result_key = _store_detection_images(image, annotated, current_user.id)
    t_store = (time.perf_counter() - t1) * 1000.0
    logger.info('TIMING_ANALYSE face_detect=%.2fms inference=%.2fms storage=%.2fms total=%.2fms', t_face, t_inf, t_store, (time.perf_counter()-t0)*1000.0)

    result = {
        'emotion': prediction['emotion'],
        'confidence': prediction['confidence'],
        'probabilities': prediction['probabilities'],
        'upload_key': upload_key,
        'result_key': result_key,
        'face_count': len(faces),
        'processing_time': processing_time,
        'detection_type': detection_type,
        'detected_at': datetime.now(timezone.utc).isoformat(),
        # Unguessable capability token for the unsaved-result preview route. It
        # lives only in this user's signed session cookie and is compared with
        # constant time, so a preview can never be reached by guessing an id.
        'preview_token': secrets.token_urlsafe(32),
        '_timing': {
            'face_detect_ms': t_face,
            'inference_ms': t_inf,
            'storage_ms': t_store,
        }
    }
    return result, None


# =============================================================================
# Image upload detection
# =============================================================================
@detection_bp.route('/detect/upload', methods=['POST'])
@active_user_required
def detect_upload():
    """Handle a multipart image upload and run detection."""
    if 'image' not in request.files:
        flash('No image file was uploaded.', 'danger')
        return redirect(url_for('detection.detect_page'))

    uploaded = request.files['image']
    if not uploaded or not uploaded.filename:
        flash('No image file was selected.', 'danger')
        return redirect(url_for('detection.detect_page'))

    # Never trust the client filename for the stored path; only its extension is
    # consulted, and only to validate the declared type.
    suffix = safe_extension(uploaded.filename)
    if suffix is None:
        flash('Invalid file type. Please upload a JPG, JPEG, or PNG image.', 'danger')
        return redirect(url_for('detection.detect_page'))

    declared_type = (uploaded.mimetype or '').lower()
    if declared_type and declared_type not in Config.ALLOWED_MIME_TYPES:
        flash('Invalid file type. Please upload a JPG, JPEG, or PNG image.', 'danger')
        return redirect(url_for('detection.detect_page'))

    file_bytes = uploaded.read()
    if not file_bytes:
        flash('The uploaded file was empty.', 'danger')
        return redirect(url_for('detection.detect_page'))

    image, error = decode_image_payload(
        base64.b64encode(file_bytes).decode('ascii')
    )
    if image is None:
        flash(error or 'The uploaded file could not be read.', 'danger')
        return redirect(url_for('detection.detect_page'))

    # Verify the decoded pixels really are one of the allowed formats. A file
    # renamed to .jpg but actually holding something else is rejected here.
    actual_type = _sniff_image_type(file_bytes)
    if actual_type not in Config.ALLOWED_MIME_TYPES:
        flash('The file is not a valid JPEG or PNG image.', 'danger')
        return redirect(url_for('detection.detect_page'))

    # The upload has now been fully validated, so only then is the model
    # consulted. An invalid file is rejected even when the model is broken, and
    # an expensive model load is never paid for a request that cannot succeed.
    if not emotion_predictor.ensure_loaded():
        flash('Emotion model unavailable. ' + emotion_predictor.get_status_message(), 'danger')
        return redirect(url_for('detection.detect_page'))

    try:
        result, error = _analyse_and_store(image, 'image')
    except ModelUnavailableError as exc:
        logger.warning('Prediction refused: %s', exc)
        flash('Emotion model unavailable. ' + emotion_predictor.get_status_message(), 'danger')
        return redirect(url_for('detection.detect_page'))
    except StorageError as exc:
        logger.error('Storage failure on upload detection: %s', exc)
        flash('The image could not be saved. Please try again.', 'danger')
        return redirect(url_for('detection.detect_page'))
    except Exception as exc:
        logger.exception('Upload detection failed')
        flash('An error occurred while processing the image. Please try again.', 'danger')
        return redirect(url_for('detection.detect_page'))

    if result is None:
        flash(error, 'warning')
        return redirect(url_for('detection.detect_page'))

    session['detection_result'] = result
    return redirect(url_for('detection.show_result'))


def _sniff_image_type(data):
    """
    Identify the real image format from its magic bytes.

    Used so a renamed executable or text file cannot pass the extension check.
    """
    if not data:
        return ''
    if data[:3] == b'\xff\xd8\xff':
        return 'image/jpeg'
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'image/png'
    return ''


# =============================================================================
# Camera capture detection
# =============================================================================
@detection_bp.route('/detect/capture', methods=['POST'])
@active_user_required
def detect_capture():
    """Handle a single browser-captured still frame (base64) and run detection."""
    t_total_start = time.perf_counter()
    t_recv = 0.0
    t_decode = 0.0
    t_analyse_total_contribution_face = 0.0  # we'll log from analyse
    t_inf = 0.0
    t_store = 0.0
    t_save_db = 0.0
    t_resp = 0.0

    t_model_check = time.perf_counter()
    if not emotion_predictor.ensure_loaded():
        logger.info('TIMING_CAPTURE recv=%.2fms decode=%.2fms face_detect=%.2fms inference=%.2fms storage=%.2fms save_db=%.2fms resp=%.2fms total_server=%.2fms',
                    0, 0, 0, 0, 0, 0, 0, (time.perf_counter()-t_total_start)*1000.0)
        return _model_unavailable_response()

    try:
        t1 = time.perf_counter()
        data = request.get_json(silent=True) or {}
        t_recv = (time.perf_counter() - t1) * 1000.0

        t1 = time.perf_counter()
        image, error = decode_image_payload(data.get('image'))
        t_decode = (time.perf_counter() - t1) * 1000.0
        if image is None:
            t_resp = (time.perf_counter() - (t1)) * 0.0  # not needed
            logger.info('TIMING_CAPTURE recv=%.2fms decode=%.2fms face_detect=%.2fms inference=%.2fms storage=%.2fms save_db=%.2fms resp=%.2fms total_server=%.2fms',
                        t_recv, t_decode, 0, 0, 0, 0, 0, (time.perf_counter()-t_total_start)*1000.0)
            return jsonify({'success': False, 'error': error}), 400

        t1 = time.perf_counter()
        result, error = _analyse_and_store(image, 'camera')
        t_analyse = (time.perf_counter() - t1) * 1000.0
        timing = result.get('_timing') if result else {}
        t_face = timing.get('face_detect_ms', 0.0) if timing else 0.0
        t_inf = timing.get('inference_ms', 0.0) if timing else 0.0
        t_store = timing.get('storage_ms', 0.0) if timing else 0.0
        if result is None:
            logger.info('TIMING_CAPTURE recv=%.2fms decode=%.2fms face_detect=%.2fms inference=%.2fms storage=%.2fms save_db=%.2fms resp=%.2fms total_server=%.2fms',
                        t_recv, t_decode, t_face, t_inf, t_store, 0.0, 0.0, (time.perf_counter()-t_total_start)*1000.0)
            return jsonify({'success': False, 'error': error}), 200

        # Remove internal timing before storing in session
        result.pop('_timing', None)
        session['detection_result'] = result
        t2 = time.perf_counter()
        resp = jsonify({'success': True, 'redirect': url_for('detection.show_result')})
        t_resp = (time.perf_counter() - t2) * 1000.0
        logger.info('TIMING_CAPTURE recv=%.2fms decode=%.2fms face_detect=%.2fms inference=%.2fms storage=%.2fms save_db=%.2fms resp=%.2fms total_server=%.2fms',
                    t_recv, t_decode, t_face, t_inf, t_store, 0.0, t_resp, (time.perf_counter()-t_total_start)*1000.0)
        return resp

    except ModelUnavailableError:
        return _model_unavailable_response()
    except StorageError as exc:
        logger.error('Storage failure on capture detection: %s', exc)
        return jsonify({
            'success': False,
            'error': 'The image could not be saved. Please try again.',
        }), 500
    except Exception:
        logger.exception('Capture detection failed')
        return jsonify({
            'success': False,
            'error': 'An error occurred while processing the image.',
        }), 500


# =============================================================================
# Result pages
# =============================================================================
@detection_bp.route('/result')
@active_user_required
def show_result():
    """Display the most recent unsaved detection result (PRG pattern)."""
    result = session.get('detection_result')
    if not result:
        flash('No detection result to display. Please perform a detection first.', 'info')
        return redirect(url_for('detection.detect_page'))

    # Images are streamed through the authorised route rather than a public URL.
    preview_key = result.get('result_key') or result.get('upload_key')
    result['image_available'] = bool(is_valid_object_key(preview_key or ''))
    if result['image_available'] and result.get('preview_token'):
        result['image_url'] = url_for(
            'detection.result_preview', preview_token=result['preview_token']
        )

    return render_template('result.html', result=result)


@detection_bp.route('/detection/preview/<preview_token>')
@active_user_required
def result_preview(preview_token):
    """
    Stream the image for the *unsaved* result held in this session.

    A saved result is served by /detection/<id>/image instead. This route exists
    because the unsaved result has no database row yet, so there is no id to
    authorise against. Access therefore requires all three of:

      1. an authenticated, active account (the decorator),
      2. a pending result in *this* session's signed cookie,
      3. the exact unguessable preview token minted when the detection ran.

    The comparison is constant time, and the storage key is re-validated for
    ownership before anything is read.
    """
    result = session.get('detection_result')
    if not result:
        abort(404)

    expected = result.get('preview_token')
    if not expected or not hmac_compare(expected, preview_token):
        abort(403)

    key = result.get('result_key') or result.get('upload_key')
    if not is_valid_object_key(key or ''):
        abort(404)
    if not is_object_key_owned_by(key, current_user.id):
        abort(403)

    payload = get_storage().load(key)
    if not payload:
        abort(404)

    data, content_type = payload
    return send_file(
        bytes(data),
        mimetype=content_type,
        max_age=0,
        conditional=False,
        download_name='detection-preview.jpg',
    )


@detection_bp.route('/result/<int:detection_id>')
@active_user_required
def view_saved_result(detection_id):
    """
    View a saved detection result.

    The lookup is filtered by ``user_id=current_user.id``, so editing the id in
    the URL cannot expose another user's record - the request simply 403s.
    """
    detection = Detection.query.filter_by(
        id=detection_id, user_id=current_user.id
    ).first()

    if detection is None:
        # Distinguish "does not exist" from "belongs to someone else" without
        # leaking the latter: both are refused.
        exists = db.session.get(Detection, detection_id)
        if exists is not None:
            abort(403)
        flash('Detection result not found.', 'warning')
        return redirect(url_for('dashboard.profile'))

    result = {
        'emotion': detection.emotion,
        'confidence': detection.confidence,
        'probabilities': {},
        'upload_path': detection.image_path,
        'result_path': detection.processed_image_path or detection.image_path,
        'image_available': is_valid_object_key(
            detection.processed_image_path or detection.image_path or ''
        ),
        'image_url': url_for('detection.detection_image', detection_id=detection.id),
        'face_count': detection.face_count,
        'processing_time': detection.processing_time,
        'detection_type': detection.detection_type,
        'detected_at': detection.detected_at.isoformat() if detection.detected_at else None,
        'saved': True,
        'detection_id': detection.id,
    }

    return render_template('result.html', result=result)


@detection_bp.route('/detection/<int:detection_id>/image')
@active_user_required
def detection_image(detection_id):
    """
    Stream a detection image to its owner (or an admin).

    This is the only way an image is ever rendered. There is no public
    ``/uploads/<arbitrary-file>`` endpoint.

    Rows written before the storage move still hold a legacy
    ``static/uploads/...`` path. Those are served from disk through
    ``local_fallback_path``, which confines resolution to the ``static``
    directory, so historical results keep working without re-uploading.
    """
    detection = db.session.get(Detection, detection_id)
    if detection is None:
        abort(404)

    # Admins may view any image; a regular user only their own.
    owns_object(detection.user_id)

    key = detection.processed_image_path or detection.image_path

    if is_valid_object_key(key or ''):
        payload = get_storage().load(key)
        if not payload:
            abort(404)
        data, content_type = payload
        return send_file(
            bytes(data),
            mimetype=content_type,
            max_age=0,
            conditional=False,
            download_name=f'detection-{detection_id}.jpg',
        )

    legacy = local_fallback_path(key)
    if legacy is None:
        abort(404)

    return send_file(
        str(legacy),
        mimetype=_sniff_image_type(legacy.read_bytes()) or 'application/octet-stream',
        max_age=0,
        conditional=False,
        download_name=f'detection-{detection_id}.jpg',
    )


# =============================================================================
# Save / delete results
# =============================================================================
@detection_bp.route('/result/save', methods=['POST'])
@active_user_required
def save_result():
    """Persist the current session result as a Detection row."""
    result = session.get('detection_result')
    if not result:
        flash('No detection result to save.', 'warning')
        return redirect(url_for('detection.detect_page'))

    required = ('emotion', 'confidence', 'detection_type', 'detected_at')
    if not all(field in result for field in required):
        flash('Invalid detection data. Cannot save.', 'danger')
        return redirect(url_for('detection.detect_page'))

    # Only labels the configured model can actually emit are accepted. This is
    # what prevents an arbitrary string being written to the database.
    if result['emotion'] not in Config.EMOTION_LABELS:
        logger.error('Refusing to save unknown emotion label: %r', result['emotion'])
        flash('Invalid emotion label. Cannot save.', 'danger')
        return redirect(url_for('detection.detect_page'))

    try:
        detected_at = datetime.fromisoformat(result['detected_at'])
        if detected_at.tzinfo is None:
            detected_at = detected_at.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        detected_at = datetime.now(timezone.utc)

    try:
        detection = Detection(
            user_id=current_user.id,
            detection_type=result['detection_type'],
            emotion=result['emotion'],
            confidence=float(result['confidence']),
            image_path=result.get('upload_key'),
            processed_image_path=result.get('result_key'),
            detected_at=detected_at,
            face_count=result.get('face_count', 1),
            processing_time=result.get('processing_time'),
        )
        db.session.add(detection)
        db.session.commit()

        # Clearing the session prevents a duplicate save on page refresh.
        session.pop('detection_result', None)

        log_activity(
            ActivityType.IMAGE_DETECTION,
            details=f'detection_id={detection.id}; type={detection.detection_type}',
        )

        flash('Detection result saved successfully!', 'success')
        return redirect(url_for('detection.view_saved_result', detection_id=detection.id))

    except Exception as exc:
        db.session.rollback()
        logger.exception('Could not save detection')
        flash('Failed to save the detection result.', 'danger')
        return redirect(url_for('detection.show_result'))


@detection_bp.route('/detect/delete/<int:detection_id>', methods=['POST'])
@active_user_required
def delete_result(detection_id):
    """Delete one of the current user's detections and its stored images."""
    detection = Detection.query.filter_by(
        id=detection_id, user_id=current_user.id
    ).first()

    if detection is None:
        exists = db.session.get(Detection, detection_id)
        if exists is not None:
            abort(403)
        flash('Detection result not found.', 'warning')
        return redirect(url_for('dashboard.profile'))

    owner_id = detection.user_id

    try:
        storage = get_storage()
        for key in (detection.image_path, detection.processed_image_path):
            # Ownership is re-verified against the key itself, so a tampered
            # database row still cannot cause another user's object to be deleted.
            if key and is_valid_object_key(key) and key.startswith(f'users/{owner_id}/'):
                storage.delete(key)
    except Exception as exc:
        logger.warning('Could not delete stored images for detection %s: %s', detection_id, exc)

    db.session.delete(detection)
    db.session.commit()
    flash('Detection result deleted.', 'success')
    return redirect(request.referrer or url_for('dashboard.profile'))


@detection_bp.route('/detect/clear-history', methods=['POST'])
@active_user_required
def clear_history():
    """Delete every detection belonging to the current user."""
    detections = Detection.query.filter_by(user_id=current_user.id).all()
    if not detections:
        flash('Your history is already empty.', 'info')
        return redirect(url_for('dashboard.profile'))

    try:
        storage = get_storage()
        for detection in detections:
            for key in (detection.image_path, detection.processed_image_path):
                if key and is_valid_object_key(key) and key.startswith(f'users/{current_user.id}/'):
                    try:
                        storage.delete(key)
                    except Exception as exc:
                        logger.warning('Could not delete object %s: %s', key, exc)
    except Exception as exc:
        logger.warning('Storage cleanup during history clear failed: %s', exc)

    Detection.query.filter_by(user_id=current_user.id).delete(synchronize_session=False)
    db.session.commit()

    flash(f'{len(detections)} detection(s) deleted.', 'success')
    return redirect(url_for('dashboard.profile'))


# =============================================================================
# Live camera detection
# =============================================================================
# The browser owns the webcam (navigator.mediaDevices.getUserMedia). The server
# receives one frame at a time, at a controlled interval, and never stores the
# video stream.
# =============================================================================
@detection_bp.route('/live')
@active_user_required
def live_detection_page():
    """Live detection page."""
    if not emotion_predictor.ensure_loaded():
        logger.warning('Live detection page opened without a usable model')
    return render_template(
        'live_detection.html',
        live_interval_ms=Config.LIVE_DETECTION_INTERVAL * 1000,
        max_session_seconds=Config.MAX_LIVE_SESSION_SECONDS,
    )


@detection_bp.route('/live/start-session', methods=['POST'])
@active_user_required
def start_live_session():
    """Create a LiveSession row and return the tokens the browser will use."""
    if not emotion_predictor.ensure_loaded():
        return _model_unavailable_response()

    try:
        started_at = datetime.now(timezone.utc)
        # Cryptographically random: the token authorises frame submission for
        # this session, so it must not be derivable from the user id and clock.
        token = secrets.token_urlsafe(36)

        live_session = LiveSession(
            user_id=current_user.id,
            started_at=started_at,
            created_at=started_at,
            session_token=token,
            total_detections=0,
            duration_seconds=0,
        )
        db.session.add(live_session)
        db.session.commit()

        log_activity(ActivityType.LIVE_SESSION_START, details=f'session_id={live_session.id}')

        return jsonify({
            'success': True,
            'session_id': token,
            'db_session_id': live_session.id,
            'started_at': started_at.isoformat(),
            'interval_ms': Config.LIVE_DETECTION_INTERVAL * 1000,
            'max_session_seconds': Config.MAX_LIVE_SESSION_SECONDS,
        })

    except Exception:
        db.session.rollback()
        logger.exception('Could not start live session')
        return jsonify({'success': False, 'error': 'Failed to start the session.'}), 500


@detection_bp.route('/live/process-frame', methods=['POST'])
@active_user_required
def process_live_frame():
    """
    Process a single live frame.

    Protection against accidental abuse:
      * a sliding-window rate limit per user
      * a minimum interval between processed frames for the session, enforced
        against the database so it survives a serverless cold start
      * a maximum session duration
    """
    if not emotion_predictor.ensure_loaded():
        return _model_unavailable_response()

    user_id = current_user.id

    allowed, retry_after = live_frame_limiter.allow(f'user:{user_id}')
    if not allowed:
        response = jsonify({
            'success': False,
            'error': 'Too many frames processed. Slow down the capture rate.',
            'retry_after': retry_after,
        })
        response.status_code = 429
        response.headers['Retry-After'] = str(retry_after)
        return response

    try:
        data = request.get_json(silent=True) or {}
        token = (data.get('session_id') or '').strip()
        db_session_id = data.get('db_session_id')

        if not db_session_id:
            return jsonify({'success': False, 'error': 'No session ID supplied.'}), 400

        # Session must belong to this user. Admins are not accepted here: a live
        # session always has exactly one owner.
        live_session = LiveSession.query.filter_by(
            id=db_session_id, user_id=user_id
        ).first()

        if live_session is None:
            return jsonify({'success': False, 'error': 'Session not found.'}), 404

        if not hmac_compare(live_session.session_token, token):
            # The token is mandatory, not optional: an empty or missing value
            # must not be able to skip the check.
            return jsonify({'success': False, 'error': 'Invalid session token.'}), 403

        now = datetime.now(timezone.utc)

        # --- Maximum session duration ---
        started_at = live_session.started_at
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        elapsed = (now - started_at).total_seconds()
        if elapsed > Config.MAX_LIVE_SESSION_SECONDS:
            return jsonify({
                'success': False,
                'error': 'This live session has reached its maximum duration. '
                         'Please start a new session.',
                'session_expired': True,
            }), 400

        # --- Minimum frame interval, enforced against the database ---
        last_detection = (
            Detection.query.filter_by(
                user_id=user_id, session_id=live_session.session_token or token
            )
            .order_by(Detection.detected_at.desc())
            .first()
        )
        if last_detection is not None:
            previous = last_detection.detected_at
            if previous.tzinfo is None:
                previous = previous.replace(tzinfo=timezone.utc)
            since_last = (now - previous).total_seconds()
            if since_last < Config.LIVE_DETECTION_INTERVAL:
                wait = round(Config.LIVE_DETECTION_INTERVAL - since_last, 2)
                return jsonify({
                    'success': True,
                    'throttled': True,
                    'retry_in': wait,
                    'message': 'Frame skipped: minimum interval not reached.',
                })

        image, error = decode_image_payload(data.get('image'))
        if image is None:
            return jsonify({'success': False, 'error': error}), 400

        faces = detect_faces(image)
        if not faces:
            # Not an error: the user may simply have looked away.
            return jsonify({
                'success': True,
                'face_detected': False,
                'message': 'No face detected',
            })

        primary_face = get_largest_face(faces)
        face_crop = crop_face(image, primary_face)
        if face_crop is None or face_crop.size == 0:
            return jsonify({'success': True, 'face_detected': False})

        started = time.perf_counter()
        # Diagnostic context for the PREDICT_PROBS log line only.
        prediction = emotion_predictor.predict(
            face_crop,
            face_context=f'live: largest of {len(faces)} face(s)',
        )
        processing_time = round(time.perf_counter() - started, 3)

        # Persist the detection EVENT only. The frame itself is not stored.
        detection = Detection(
            user_id=user_id,
            detection_type='camera',
            emotion=prediction['emotion'],
            confidence=prediction['confidence'],
            detected_at=now,
            session_id=live_session.session_token or token,
            face_count=len(faces),
            processing_time=processing_time,
        )
        db.session.add(detection)

        live_session.total_detections = (live_session.total_detections or 0) + 1
        live_session.duration_seconds = int(elapsed)
        db.session.commit()

        x, y, w, h = primary_face
        return jsonify({
            'success': True,
            'face_detected': True,
            'emotion': prediction['emotion'],
            'confidence': prediction['confidence'],
            'probabilities': prediction['probabilities'],
            'processing_time': processing_time,
            'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
        })

    except ModelUnavailableError:
        return _model_unavailable_response()
    except Exception:
        db.session.rollback()
        logger.exception('Live frame processing failed')
        return jsonify({
            'success': False,
            'error': 'An error occurred while processing the frame.',
        }), 500


@detection_bp.route('/live/end-session', methods=['POST'])
@active_user_required
def end_live_session():
    """Close a live session and store its summary statistics."""
    try:
        data = request.get_json(silent=True) or {}
        db_session_id = data.get('db_session_id')
        token = (data.get('session_id') or '').strip()

        if not db_session_id:
            return jsonify({'success': False, 'error': 'No session ID supplied.'}), 400

        live_session = LiveSession.query.filter_by(
            id=db_session_id, user_id=current_user.id
        ).first()

        if live_session is None:
            return jsonify({'success': False, 'error': 'Session not found.'}), 404

        if not hmac_compare(live_session.session_token, token):
            # Mandatory, matching /live/process-frame.
            return jsonify({'success': False, 'error': 'Invalid session token.'}), 403

        now = datetime.now(timezone.utc)

        started_at = live_session.started_at
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        duration = max(0, int((now - started_at).total_seconds()))

        live_session.ended_at = now
        live_session.duration_seconds = duration

        session_token = live_session.session_token or token

        # Aggregate with database functions rather than loading every row.
        from sqlalchemy import func

        stats = (
            db.session.query(
                func.count(Detection.id),
                func.coalesce(func.avg(Detection.confidence), 0),
            )
            .filter(
                Detection.user_id == current_user.id,
                Detection.session_id == session_token,
            )
            .one()
        )
        total_detections = stats[0] or 0
        live_session.total_detections = total_detections

        if total_detections:
            live_session.average_confidence = round(float(stats[1]), 2)

            dominant = (
                db.session.query(Detection.emotion, func.count(Detection.id))
                .filter(
                    Detection.user_id == current_user.id,
                    Detection.session_id == session_token,
                )
                .group_by(Detection.emotion)
                .order_by(func.count(Detection.id).desc())
                .limit(1)
                .first()
            )
            if dominant:
                live_session.dominant_emotion = dominant[0]

        db.session.commit()

        log_activity(ActivityType.LIVE_SESSION_END, details=f'session_id={live_session.id}')

        return jsonify({
            'success': True,
            'summary': {
                'duration_seconds': live_session.duration_seconds,
                'total_detections': live_session.total_detections,
                'dominant_emotion': live_session.dominant_emotion,
                'average_confidence': live_session.average_confidence,
            },
        })

    except Exception:
        db.session.rollback()
        logger.exception('Could not end live session')
        return jsonify({
            'success': False,
            'error': 'An error occurred while ending the session.',
        }), 500
