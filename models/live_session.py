# =============================================================================
# Live Session Model
# =============================================================================
# Summary of a live webcam detection session.
#
# All timestamps are stored as timezone-aware UTC. PostgreSQL columns are
# TIMESTAMP WITHOUT TIME ZONE (portable across SQLite and PostgreSQL) and are
# always written with an explicit UTC offset by the application, so values are
# unambiguous. Conversion to the viewer's local time happens at display time in
# the templates only.
# =============================================================================

from datetime import datetime, timezone

from utils.database import db


def utcnow():
    return datetime.now(timezone.utc)


class LiveSession(db.Model):
    """Summary of one live webcam emotion detection session."""

    __tablename__ = 'live_sessions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'),
                        nullable=False, index=True)

    started_at = db.Column(db.DateTime, nullable=False,
                           default=utcnow, index=True)
    ended_at = db.Column(db.DateTime, nullable=True)
    duration_seconds = db.Column(db.Integer, nullable=True, default=0)

    dominant_emotion = db.Column(db.String(50), nullable=True)
    average_confidence = db.Column(db.Float, nullable=True)
    total_detections = db.Column(db.Integer, nullable=True, default=0)

    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    # Client-side token that ties Detection rows to this session. Indexed so a
    # session summary never needs a full table scan.
    session_token = db.Column(db.String(100), nullable=True, index=True)

    __table_args__ = (
        db.Index('ix_live_sessions_user_started', 'user_id', 'started_at'),
    )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    @property
    def is_active_session(self):
        return self.ended_at is None

    @property
    def duration_minutes(self):
        if not self.duration_seconds:
            return 0.0
        return round(self.duration_seconds / 60.0, 1)

    def __repr__(self):
        return f'<LiveSession {self.id}: {self.dominant_emotion} ({self.total_detections} events)>'
