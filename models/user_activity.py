# =============================================================================
# User Activity Model
# =============================================================================
# Audit trail of notable account and detection events.
#
# PRIVACY: this table intentionally records the client IP address and User-Agent
# string for security auditing (abuse detection, session investigation). Those
# two fields are personal data. See the PRIVACY section of README.md.
#
# This table must never store passwords, password hashes, session cookies,
# CSRF tokens, API keys or any Supabase credential.
# =============================================================================

from datetime import datetime, timezone

from utils.database import db


def utcnow():
    return datetime.now(timezone.utc)


class ActivityType:
    """Canonical activity type constants."""

    REGISTER = 'REGISTER'
    LOGIN = 'LOGIN'
    LOGOUT = 'LOGOUT'
    IMAGE_DETECTION = 'IMAGE_DETECTION'
    LIVE_SESSION_START = 'LIVE_SESSION_START'
    LIVE_SESSION_END = 'LIVE_SESSION_END'
    EXPORT = 'EXPORT'
    PROFILE_UPDATE = 'PROFILE_UPDATE'
    PASSWORD_CHANGE = 'PASSWORD_CHANGE'
    ACCOUNT_DEACTIVATED = 'ACCOUNT_DEACTIVATED'
    ACCOUNT_REACTIVATED = 'ACCOUNT_REACTIVATED'

    # Admin actions
    ADMIN_DEACTIVATE_USER = 'ADMIN_DEACTIVATE_USER'
    ADMIN_ACTIVATE_USER = 'ADMIN_ACTIVATE_USER'
    ADMIN_DELETE_USER = 'ADMIN_DELETE_USER'
    ADMIN_CHANGE_ROLE = 'ADMIN_CHANGE_ROLE'
    ADMIN_EXPORT = 'ADMIN_EXPORT'

    ALL = (
        REGISTER, LOGIN, LOGOUT, IMAGE_DETECTION, LIVE_SESSION_START,
        LIVE_SESSION_END, EXPORT, PROFILE_UPDATE, PASSWORD_CHANGE,
        ACCOUNT_DEACTIVATED, ACCOUNT_REACTIVATED, ADMIN_DEACTIVATE_USER,
        ADMIN_ACTIVATE_USER, ADMIN_DELETE_USER, ADMIN_CHANGE_ROLE,
        ADMIN_EXPORT,
    )


class UserActivity(db.Model):
    """A single audited action."""

    __tablename__ = 'user_activities'

    id = db.Column(db.Integer, primary_key=True)

    # The user the event belongs to. For admin actions this is the *acting*
    # admin, so the trail always shows who did what.
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'),
                        nullable=True, index=True)

    activity_type = db.Column(db.String(50), nullable=False, index=True)

    # Optional short context, e.g. "target_user_id=12" or "type=image".
    # Never used for credentials.
    details = db.Column(db.String(500), nullable=True)

    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.String(500), nullable=True)

    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)

    user = db.relationship(
        'User',
        backref=db.backref('activities', lazy='dynamic', cascade='all, delete-orphan'),
    )

    __table_args__ = (
        db.Index('ix_user_activities_user_created', 'user_id', 'created_at'),
    )

    @property
    def username(self):
        return self.user.username if self.user else 'Deleted user'

    def __repr__(self):
        return f'<UserActivity {self.id}: {self.activity_type}>'
