# =============================================================================
# User Model
# =============================================================================
# Stores registered user accounts with hashed passwords.
# Uses Werkzeug for secure password hashing (never stores plain-text).
# Integrates with Flask-Login via UserMixin for session management.
# =============================================================================

from datetime import datetime, timezone
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from utils.database import db


class User(UserMixin, db.Model):
    """Registered user account."""

    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    # Relationships — cascade delete so removing a user removes their data
    detections = db.relationship('Detection', backref='user', lazy='dynamic',
                                 cascade='all, delete-orphan')
    live_sessions = db.relationship('LiveSession', backref='user', lazy='dynamic',
                                    cascade='all, delete-orphan')

    def set_password(self, password):
        """Hash and store the password. Never stores plain-text."""
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')

    def check_password(self, password):
        """Verify a plain-text password against the stored hash."""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'
