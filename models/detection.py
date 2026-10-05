# =============================================================================
# Detection Model
# =============================================================================
# One row per processed detection event.
#
# `image_path` and `processed_image_path` hold *storage keys* (for example
# ``users/12/results/<uuid>.jpg``), never absolute local paths, so the same row
# is valid in SQLite development and in Supabase Storage production.
# =============================================================================

from datetime import datetime, timezone

from utils.database import db

# A storage key is a short, server-generated string. 500 leaves plenty of room.
STORAGE_KEY_LENGTH = 500


class Detection(db.Model):
    """A single emotion detection result."""

    __tablename__ = 'detections'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'),
                        nullable=False, index=True)

    # 'image' for uploads / captures, 'camera' for live detection events
    detection_type = db.Column(db.String(20), nullable=False, default='image', index=True)

    # Predicted emotion label, e.g. 'Happy'
    emotion = db.Column(db.String(50), nullable=False, index=True)

    # Confidence percentage, e.g. 94.27
    confidence = db.Column(db.Float, nullable=False)

    # Storage key for the original image
    image_path = db.Column(db.String(STORAGE_KEY_LENGTH), nullable=True)

    # Storage key for the annotated/boxed result image
    processed_image_path = db.Column(db.String(STORAGE_KEY_LENGTH), nullable=True)

    detected_at = db.Column(db.DateTime, nullable=False,
                            default=lambda: datetime.now(timezone.utc), index=True)

    # Client-side session token for live detection grouping
    session_id = db.Column(db.String(100), nullable=True, index=True)

    # Number of faces found in the frame
    face_count = db.Column(db.Integer, nullable=True, default=1)

    # Server-side processing time in seconds
    processing_time = db.Column(db.Float, nullable=True)

    __table_args__ = (
        db.Index('ix_detections_user_detected', 'user_id', 'detected_at'),
        db.Index('ix_detections_type_emotion', 'detection_type', 'emotion'),
    )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    @property
    def is_live(self):
        return self.detection_type == 'camera'

    @property
    def has_processed_image(self):
        return bool(self.processed_image_path)

    def __repr__(self):
        pct = f'{self.confidence:.1f}%' if self.confidence is not None else 'n/a'
        return f'<Detection {self.id}: {self.emotion} ({pct})>'
