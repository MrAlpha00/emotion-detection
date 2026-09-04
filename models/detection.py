# =============================================================================
# Detection Model
# =============================================================================
# Stores individual emotion detection results (both image upload and camera).
# Each record links to a user and captures the detected emotion, confidence,
# image path, timestamp, and optional session information for live detections.
# =============================================================================

from datetime import datetime, timezone
from utils.database import db


class Detection(db.Model):
    """A single emotion detection result."""

    __tablename__ = 'detections'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)

    # 'image' for uploaded/captured images, 'camera' for live detection frames
    detection_type = db.Column(db.String(20), nullable=False, default='image')

    # Predicted emotion label (e.g., 'Happy', 'Sad')
    emotion = db.Column(db.String(50), nullable=False)

    # Confidence score as a percentage (e.g., 94.27)
    confidence = db.Column(db.Float, nullable=False)

    # Path to the stored image (relative to static folder)
    image_path = db.Column(db.String(500), nullable=True)

    # Server-side timestamp of when the detection was performed
    detected_at = db.Column(db.DateTime, nullable=False,
                            default=lambda: datetime.now(timezone.utc))

    # For live detections: links to a LiveSession
    session_id = db.Column(db.String(100), nullable=True, index=True)

    # Number of faces detected in the image
    face_count = db.Column(db.Integer, nullable=True, default=1)

    # Time taken to process the image (in seconds)
    processing_time = db.Column(db.Float, nullable=True)

    def __repr__(self):
        return f'<Detection {self.id}: {self.emotion} ({self.confidence:.1f}%)>'
