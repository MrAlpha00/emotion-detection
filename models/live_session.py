# =============================================================================
# LiveSession Model
# =============================================================================
# Stores summary information for live webcam detection sessions.
# Created when a user stops a live detection session, capturing duration,
# dominant emotion, average confidence, and total detections count.
# Individual frame detections are stored in the Detection table.
# =============================================================================

from datetime import datetime, timezone
from utils.database import db


class LiveSession(db.Model):
    """Summary of a live webcam emotion detection session."""

    __tablename__ = 'live_sessions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)

    # Session timing
    started_at = db.Column(db.DateTime, nullable=False)
    ended_at = db.Column(db.DateTime, nullable=True)
    duration_seconds = db.Column(db.Integer, nullable=True, default=0)

    # Session statistics (calculated when session ends)
    dominant_emotion = db.Column(db.String(50), nullable=True)
    average_confidence = db.Column(db.Float, nullable=True)
    total_detections = db.Column(db.Integer, nullable=True, default=0)

    # Record creation timestamp
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f'<LiveSession {self.id}: {self.dominant_emotion} ({self.total_detections} detections)>'
