# =============================================================================
# Detection Routes
# =============================================================================
# Handles image upload, camera capture, face detection, emotion prediction,
# result display, live camera detection, and result management.
# All prediction routes check model availability before proceeding.
# =============================================================================

import os
import uuid
import time
import base64
import cv2
import numpy as np
from datetime import datetime, timezone
from flask import (Blueprint, render_template, redirect, url_for,
                   flash, request, jsonify, session)
from flask_login import login_required, current_user
from config import Config
from utils.database import db
from utils.face_detector import (detect_faces, crop_face, get_largest_face,
                                  draw_bounding_boxes, draw_emotion_boxes)
from utils.emotion_predictor import emotion_predictor, ModelUnavailableError
from models.detection import Detection
from models.live_session import LiveSession

detection_bp = Blueprint('detection', __name__)


# =============================================================================
# Helper Functions
# =============================================================================

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS)


def generate_unique_filename(extension='jpg'):
    """Generate a unique filename using UUID to avoid collisions."""
    return f"{uuid.uuid4().hex}.{extension}"


def save_image(image, folder, filename=None):
    """
    Save an OpenCV image to disk.

    Args:
        image: OpenCV image (numpy array).
        folder: Directory to save in.
        filename: Optional filename; generated if not provided.

    Returns:
        The relative path from static/ for use in templates.
    """
    if filename is None:
        filename = generate_unique_filename()

    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, image)

    # Return path relative to 'static/' for url_for('static', ...)
    static_dir = os.path.join(Config.BASE_DIR, 'static')
    rel_path = os.path.relpath(filepath, static_dir)
    return rel_path.replace('\\', '/')


# =============================================================================
# Detection Page
# =============================================================================

@detection_bp.route('/detect')
@login_required
def detect_page():
    """Show the face detection page (upload or camera capture)."""
    return render_template('face_detection.html')


# =============================================================================
# Image Upload Detection
# =============================================================================

@detection_bp.route('/detect/upload', methods=['POST'])
@login_required
def detect_upload():
    """Handle image file upload and perform emotion detection."""

    # --- Check model availability ---
    if not emotion_predictor.is_available():
        flash('Emotion model is not available. ' + emotion_predictor.get_status_message(), 'danger')
        return redirect(url_for('detection.detect_page'))

    # --- Validate file ---
    if 'image' not in request.files:
        flash('No image file was uploaded.', 'danger')
        return redirect(url_for('detection.detect_page'))

    file = request.files['image']
    if file.filename == '':
        flash('No image file was selected.', 'danger')
        return redirect(url_for('detection.detect_page'))

    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload a JPG, JPEG, or PNG image.', 'danger')
        return redirect(url_for('detection.detect_page'))

    try:
        start_time = time.time()

        # Read the uploaded image
        file_bytes = file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            flash('The uploaded file is not a valid image or is corrupt.', 'danger')
            return redirect(url_for('detection.detect_page'))

        # --- Face Detection ---
        faces = detect_faces(image)

        if len(faces) == 0:
            flash('No face was detected in the uploaded image. Please try a clearer photo.', 'warning')
            return redirect(url_for('detection.detect_page'))

        # Use the largest face as the primary face
        primary_face = get_largest_face(faces)
        face_crop = crop_face(image, primary_face)

        if face_crop is None or face_crop.size == 0:
            flash('Failed to crop the detected face. Please try another image.', 'danger')
            return redirect(url_for('detection.detect_page'))

        # --- Emotion Prediction ---
        try:
            result = emotion_predictor.predict(face_crop)
        except ModelUnavailableError as e:
            flash(str(e), 'danger')
            return redirect(url_for('detection.detect_page'))

        processing_time = round(time.time() - start_time, 3)

        # --- Save images ---
        # Save original uploaded image
        upload_filename = generate_unique_filename('jpg')
        upload_path = save_image(image, Config.UPLOAD_FOLDER, upload_filename)

        # Save processed image with bounding boxes and emotion labels
        annotated = draw_emotion_boxes(
            image, [primary_face],
            [result['emotion']], [result['confidence']]
        )
        result_filename = f"result_{upload_filename}"
        result_path = save_image(annotated, Config.RESULTS_FOLDER, result_filename)

        # Store result in session for the result page (PRG pattern)
        session['detection_result'] = {
            'emotion': result['emotion'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'upload_path': upload_path,
            'result_path': result_path,
            'face_count': len(faces),
            'processing_time': processing_time,
            'detection_type': 'image',
            'detected_at': datetime.now(timezone.utc).isoformat()
        }

        return redirect(url_for('detection.show_result'))

    except Exception as e:
        flash(f'An error occurred while processing the image: {str(e)}', 'danger')
        print(f"[Detection] Upload error: {e}")
        return redirect(url_for('detection.detect_page'))


# =============================================================================
# Camera Capture Detection
# =============================================================================

@detection_bp.route('/detect/capture', methods=['POST'])
@login_required
def detect_capture():
    """Handle camera-captured image (base64) and perform emotion detection."""

    # --- Check model availability ---
    if not emotion_predictor.is_available():
        return jsonify({
            'success': False,
            'error': 'Emotion model is not available. ' + emotion_predictor.get_status_message()
        }), 503

    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data received.'}), 400

        # Decode base64 image
        image_data = data['image']
        # Remove the data URL prefix if present (e.g., 'data:image/png;base64,')
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'success': False, 'error': 'Invalid image data.'}), 400

        start_time = time.time()

        # --- Face Detection ---
        faces = detect_faces(image)

        if len(faces) == 0:
            return jsonify({
                'success': False,
                'error': 'No face detected. Please ensure your face is clearly visible.'
            }), 200

        # Use the largest face
        primary_face = get_largest_face(faces)
        face_crop = crop_face(image, primary_face)

        if face_crop is None or face_crop.size == 0:
            return jsonify({'success': False, 'error': 'Failed to crop the detected face.'}), 200

        # --- Emotion Prediction ---
        try:
            result = emotion_predictor.predict(face_crop)
        except ModelUnavailableError as e:
            return jsonify({'success': False, 'error': str(e)}), 503

        processing_time = round(time.time() - start_time, 3)

        # --- Save images ---
        upload_filename = generate_unique_filename('jpg')
        upload_path = save_image(image, Config.UPLOAD_FOLDER, upload_filename)

        annotated = draw_emotion_boxes(
            image, [primary_face],
            [result['emotion']], [result['confidence']]
        )
        result_filename = f"result_{upload_filename}"
        result_path = save_image(annotated, Config.RESULTS_FOLDER, result_filename)

        # Store in session for the result page
        session['detection_result'] = {
            'emotion': result['emotion'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'upload_path': upload_path,
            'result_path': result_path,
            'face_count': len(faces),
            'processing_time': processing_time,
            'detection_type': 'camera',
            'detected_at': datetime.now(timezone.utc).isoformat()
        }

        return jsonify({
            'success': True,
            'redirect': url_for('detection.show_result')
        })

    except Exception as e:
        print(f"[Detection] Capture error: {e}")
        return jsonify({'success': False, 'error': 'An error occurred while processing.'}), 500


# =============================================================================
# Result Page
# =============================================================================

@detection_bp.route('/result')
@login_required
def show_result():
    """Display the detection result from the session (PRG pattern)."""
    result = session.get('detection_result')

    if not result:
        flash('No detection result to display. Please perform a detection first.', 'info')
        return redirect(url_for('detection.detect_page'))

    return render_template('result.html', result=result)


@detection_bp.route('/result/<int:detection_id>')
@login_required
def view_saved_result(detection_id):
    """View a previously saved detection result."""
    detection = Detection.query.filter_by(
        id=detection_id, user_id=current_user.id
    ).first()

    if not detection:
        flash('Detection result not found.', 'warning')
        return redirect(url_for('dashboard.profile'))

    # Convert saved detection to result format for the template
    result = {
        'emotion': detection.emotion,
        'confidence': detection.confidence,
        'probabilities': {},  # Not stored individually; show only main result
        'upload_path': detection.image_path,
        'result_path': detection.image_path,
        'face_count': detection.face_count,
        'processing_time': detection.processing_time,
        'detection_type': detection.detection_type,
        'detected_at': detection.detected_at.isoformat() if detection.detected_at else None,
        'saved': True,
        'detection_id': detection.id
    }

    return render_template('result.html', result=result)


# =============================================================================
# Save Result
# =============================================================================

@detection_bp.route('/result/save', methods=['POST'])
@login_required
def save_result():
    """Save the current detection result to the database."""
    result = session.get('detection_result')

    if not result:
        flash('No detection result to save.', 'warning')
        return redirect(url_for('detection.detect_page'))

    # Validate that this is a genuine prediction (has required fields)
    required_fields = ['emotion', 'confidence', 'detection_type', 'detected_at']
    if not all(field in result for field in required_fields):
        flash('Invalid detection data. Cannot save.', 'danger')
        return redirect(url_for('detection.detect_page'))

    # Validate emotion is a known label
    if result['emotion'] not in Config.EMOTION_LABELS:
        flash('Invalid emotion label. Cannot save.', 'danger')
        return redirect(url_for('detection.detect_page'))

    try:
        detection = Detection(
            user_id=current_user.id,
            detection_type=result['detection_type'],
            emotion=result['emotion'],
            confidence=result['confidence'],
            image_path=result.get('result_path'),
            detected_at=datetime.fromisoformat(result['detected_at']),
            face_count=result.get('face_count', 1),
            processing_time=result.get('processing_time')
        )
        db.session.add(detection)
        db.session.commit()

        # Clear the session result to prevent duplicate saves on refresh
        session.pop('detection_result', None)

        flash('Detection result saved successfully!', 'success')
        return redirect(url_for('detection.view_saved_result', detection_id=detection.id))

    except Exception as e:
        db.session.rollback()
        flash('Failed to save the detection result.', 'danger')
        print(f"[Detection] Save error: {e}")
        return redirect(url_for('detection.show_result'))


# =============================================================================
# Delete Result / Clear History
# =============================================================================

@detection_bp.route('/detect/delete/<int:detection_id>', methods=['POST'])
@login_required
def delete_result(detection_id):
    """Delete a single detection result (only the owner can delete)."""
    detection = Detection.query.filter_by(
        id=detection_id, user_id=current_user.id
    ).first()

    if not detection:
        flash('Detection result not found.', 'warning')
        return redirect(url_for('dashboard.profile'))

    try:
        # Delete associated image files if they exist
        if detection.image_path:
            full_path = os.path.join(Config.BASE_DIR, 'static', detection.image_path)
            if os.path.exists(full_path):
                os.remove(full_path)

        db.session.delete(detection)
        db.session.commit()
        flash('Detection result deleted.', 'success')
    except Exception as e:
        db.session.rollback()
        flash('Failed to delete the detection result.', 'danger')
        print(f"[Detection] Delete error: {e}")

    return redirect(url_for('dashboard.profile'))


@detection_bp.route('/detect/clear-history', methods=['POST'])
@login_required
def clear_history():
    """Clear all detection results for the current user."""
    try:
        detections = Detection.query.filter_by(user_id=current_user.id).all()

        # Delete associated image files
        for detection in detections:
            if detection.image_path:
                full_path = os.path.join(Config.BASE_DIR, 'static', detection.image_path)
                if os.path.exists(full_path):
                    os.remove(full_path)

        Detection.query.filter_by(user_id=current_user.id).delete()

        # Also clear live sessions
        LiveSession.query.filter_by(user_id=current_user.id).delete()

        db.session.commit()
        flash('All detection history has been cleared.', 'success')
    except Exception as e:
        db.session.rollback()
        flash('Failed to clear history.', 'danger')
        print(f"[Detection] Clear history error: {e}")

    return redirect(url_for('dashboard.profile'))


# =============================================================================
# Live Camera Detection
# =============================================================================

@detection_bp.route('/live')
@login_required
def live_detection_page():
    """Show the live camera detection page."""
    return render_template('live_detection.html')


@detection_bp.route('/live/start-session', methods=['POST'])
@login_required
def start_live_session():
    """Create a new live detection session."""

    if not emotion_predictor.is_available():
        return jsonify({
            'success': False,
            'error': 'Emotion model is not available. ' + emotion_predictor.get_status_message()
        }), 503

    try:
        session_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)

        live_session = LiveSession(
            user_id=current_user.id,
            started_at=now
        )
        db.session.add(live_session)
        db.session.commit()

        return jsonify({
            'success': True,
            'session_id': session_id,
            'db_session_id': live_session.id,
            'started_at': now.isoformat()
        })

    except Exception as e:
        db.session.rollback()
        print(f"[Live] Start session error: {e}")
        return jsonify({'success': False, 'error': 'Failed to start session.'}), 500


@detection_bp.route('/live/process-frame', methods=['POST'])
@login_required
def process_live_frame():
    """Process a single frame from the live camera feed."""

    if not emotion_predictor.is_available():
        return jsonify({
            'success': False,
            'error': 'Emotion model is not available.'
        }), 503

    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data.'}), 400

        session_id = data.get('session_id', '')
        db_session_id = data.get('db_session_id')

        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'success': False, 'error': 'Invalid frame.'}), 400

        # Face detection
        faces = detect_faces(image)

        if len(faces) == 0:
            return jsonify({
                'success': True,
                'face_detected': False,
                'message': 'No face detected'
            })

        # Use the largest face
        primary_face = get_largest_face(faces)
        face_crop = crop_face(image, primary_face)

        if face_crop is None:
            return jsonify({'success': True, 'face_detected': False})

        # Emotion prediction
        try:
            result = emotion_predictor.predict(face_crop)
        except ModelUnavailableError:
            return jsonify({'success': False, 'error': 'Model unavailable.'}), 503

        # Save detection event to database (NOT the frame image)
        detection = Detection(
            user_id=current_user.id,
            detection_type='camera',
            emotion=result['emotion'],
            confidence=result['confidence'],
            detected_at=datetime.now(timezone.utc),
            session_id=session_id,
            face_count=len(faces)
        )
        db.session.add(detection)
        db.session.commit()

        # Return face bbox for overlay drawing
        x, y, w, h = primary_face
        return jsonify({
            'success': True,
            'face_detected': True,
            'emotion': result['emotion'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
        })

    except Exception as e:
        db.session.rollback()
        print(f"[Live] Process frame error: {e}")
        return jsonify({'success': False, 'error': 'Processing error.'}), 500


@detection_bp.route('/live/end-session', methods=['POST'])
@login_required
def end_live_session():
    """End a live detection session and calculate summary statistics."""

    try:
        data = request.get_json()
        db_session_id = data.get('db_session_id')
        session_id = data.get('session_id', '')

        if not db_session_id:
            return jsonify({'success': False, 'error': 'No session ID.'}), 400

        # Get the live session record
        live_session = LiveSession.query.filter_by(
            id=db_session_id, user_id=current_user.id
        ).first()

        if not live_session:
            return jsonify({'success': False, 'error': 'Session not found.'}), 404

        now = datetime.now(timezone.utc)
        live_session.ended_at = now

        # Calculate duration
        if live_session.started_at:
            duration = (now - live_session.started_at).total_seconds()
            live_session.duration_seconds = int(duration)

        # Calculate statistics from this session's detections
        session_detections = Detection.query.filter_by(
            user_id=current_user.id,
            session_id=session_id
        ).all()

        live_session.total_detections = len(session_detections)

        if session_detections:
            # Average confidence
            avg_conf = sum(d.confidence for d in session_detections) / len(session_detections)
            live_session.average_confidence = round(avg_conf, 2)

            # Dominant emotion (most frequent)
            emotion_counts = {}
            for d in session_detections:
                emotion_counts[d.emotion] = emotion_counts.get(d.emotion, 0) + 1
            live_session.dominant_emotion = max(emotion_counts, key=emotion_counts.get)

        db.session.commit()

        return jsonify({
            'success': True,
            'summary': {
                'duration_seconds': live_session.duration_seconds,
                'total_detections': live_session.total_detections,
                'dominant_emotion': live_session.dominant_emotion,
                'average_confidence': live_session.average_confidence
            }
        })

    except Exception as e:
        db.session.rollback()
        print(f"[Live] End session error: {e}")
        return jsonify({'success': False, 'error': 'Failed to end session.'}), 500
