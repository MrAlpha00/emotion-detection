# =============================================================================
# User Model
# =============================================================================
# Registered accounts. Passwords are stored only as Werkzeug PBKDF2 hashes.
# Roles are 'user' or 'admin'; the admin role is only ever assigned by the
# create_admin CLI script or by another authenticated admin, never by a form
# field supplied by the person registering.
# =============================================================================

from datetime import datetime, timezone

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from config import Config
from utils.database import db

_HASH_METHOD = 'pbkdf2:sha256'


def utcnow():
    """Timezone-aware UTC now. Used as the default for every timestamp."""
    return datetime.now(timezone.utc)


class User(UserMixin, db.Model):
    """Registered user account."""

    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)

    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    # --- Role and account state ---
    role = db.Column(
        db.String(20), nullable=False, default=Config.ROLE_USER, index=True
    )
    is_active = db.Column(db.Boolean, nullable=False, default=True, index=True)
    login_count = db.Column(db.Integer, nullable=False, default=0)
    last_login_at = db.Column(db.DateTime, nullable=True)

    # --- Relationships ---
    # delete-orphan + cascade means removing a user also removes their
    # detections, live sessions and activity rows, leaving no orphans on either
    # SQLite or PostgreSQL.
    detections = db.relationship(
        'Detection',
        backref='user',
        lazy='dynamic',
        cascade='all, delete-orphan',
        passive_deletes=False,
    )
    live_sessions = db.relationship(
        'LiveSession',
        backref='user',
        lazy='dynamic',
        cascade='all, delete-orphan',
        passive_deletes=False,
    )

    # -------------------------------------------------------------------------
    # Password handling
    # -------------------------------------------------------------------------
    def set_password(self, password):
        """Hash and store the password. Plain-text is never persisted."""
        if not password:
            raise ValueError('Password must not be empty')
        self.password_hash = generate_password_hash(password, method=_HASH_METHOD)

    def check_password(self, password):
        """Verify a plain-text password against the stored hash."""
        if not password or not self.password_hash:
            return False
        try:
            return check_password_hash(self.password_hash, password)
        except (ValueError, TypeError):
            return False

    # -------------------------------------------------------------------------
    # Role helpers
    # -------------------------------------------------------------------------
    @property
    def is_admin(self):
        """True when this account holds the admin role."""
        return self.role == Config.ROLE_ADMIN

    @property
    def role_label(self):
        return self.role.capitalize() if self.role else 'User'

    def has_role(self, role):
        return self.role == role

    # -------------------------------------------------------------------------
    # Flask-Login integration
    # -------------------------------------------------------------------------
    @property
    def is_authenticated(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)

    def __repr__(self):
        return f'<User {self.username} role={self.role}>'
