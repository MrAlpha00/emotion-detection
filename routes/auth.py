# =============================================================================
# Authentication Routes
# =============================================================================
# Registration, login and logout. All validation is server-side, passwords are
# hashed with Werkzeug, and notable events are written to the activity log.
#
# The 'role' column is never populated from a submitted form field. New accounts
# are always created with role='user'; the admin role can only be granted by the
# create_admin CLI script or by an authenticated admin from the admin area.
# =============================================================================

import re
from datetime import datetime, timezone

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from config import Config
from models.detection import Detection
from models.live_session import LiveSession
from models.user import User
from models.user_activity import ActivityType
from utils.database import db
from utils.security import log_activity

auth_bp = Blueprint('auth', __name__)

USERNAME_RE = re.compile(r'^[A-Za-z0-9_-]{3,80}$')
EMAIL_RE = re.compile(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
MIN_PASSWORD_LENGTH = 8


def valid_email(email):
    return bool(email) and EMAIL_RE.match(email) is not None


def valid_username(username):
    return bool(username) and USERNAME_RE.match(username) is not None


def _safe_next_url(candidate):
    """
    Only allow same-site relative redirects.

    Without this check an attacker could craft
    ``/login?next=https://evil.example`` and use the login page as an open
    redirect.
    """
    if not candidate:
        return None
    if candidate.startswith('/') and not candidate.startswith('//'):
        return candidate
    return None


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Authenticate with either an email address or a username."""
    if current_user.is_authenticated:
        return redirect(_safe_next_url(request.args.get('next')) or url_for('dashboard.home'))

    if request.method == 'POST':
        login_input = (request.form.get('login_input') or '').strip()
        password = request.form.get('password') or ''

        if not login_input or not password:
            flash('Please enter your email/username and password.', 'danger')
            return render_template('login.html')

        user = User.query.filter(
            (User.email == login_input.lower()) | (User.username == login_input)
        ).first()

        if user is None or not user.check_password(password):
            # Deliberately identical message for unknown user and wrong password
            # so the form does not confirm which accounts exist.
            flash('Invalid email/username or password.', 'danger')
            return render_template('login.html')

        if not user.is_active:
            flash(
                'This account has been deactivated. Please contact an administrator.',
                'danger',
            )
            return render_template('login.html')

        login_user(user, remember=bool(request.form.get('remember')))

        user.login_count = (user.login_count or 0) + 1
        user.last_login_at = datetime.now(timezone.utc)
        db.session.commit()

        log_activity(ActivityType.LOGIN, user=user)

        flash(f'Welcome back, {user.username}!', 'success')

        if user.role == Config.ROLE_ADMIN:
            return redirect(url_for('admin.dashboard'))

        return redirect(_safe_next_url(request.args.get('next')) or url_for('dashboard.home'))

    return render_template('login.html')


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Create a new standard (non-admin) account."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.home'))

    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''
        confirm_password = request.form.get('confirm_password') or ''

        errors = []

        if not username:
            errors.append('Username is required.')
        elif not valid_username(username):
            errors.append(
                'Username must be 3-80 characters using letters, numbers, underscores or hyphens.'
            )

        if not email:
            errors.append('Email is required.')
        elif not valid_email(email):
            errors.append('Please enter a valid email address.')

        if not password:
            errors.append('Password is required.')
        elif len(password) < MIN_PASSWORD_LENGTH:
            errors.append(f'Password must be at least {MIN_PASSWORD_LENGTH} characters long.')

        if password != confirm_password:
            errors.append('Passwords do not match.')

        if not errors and User.query.filter_by(username=username).first():
            errors.append('This username is already taken.')
        if not errors and User.query.filter_by(email=email).first():
            errors.append('An account with this email already exists.')

        if errors:
            for error in errors:
                flash(error, 'danger')
            return render_template('register.html')

        try:
            # Role is hard-coded to 'user' here on purpose: a self-registering
            # user can never grant themselves admin rights via a form field.
            new_user = User(
                username=username,
                email=email,
                role=Config.ROLE_USER,
                is_active=True,
                login_count=0,
            )
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()

            log_activity(ActivityType.REGISTER, user=new_user)

            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('auth.login'))
        except Exception as exc:
            db.session.rollback()
            # Log the detail server-side only; the user gets a generic message.
            app_log_exception('registration', exc)
            flash('An error occurred during registration. Please try again.', 'danger')

    return render_template('register.html')


@auth_bp.route('/logout')
@login_required
def logout():
    """Clear the session and record the event."""
    log_activity(ActivityType.LOGOUT)
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))


def app_log_exception(context, exc):
    """Log an exception without leaking internals to the browser."""
    import logging

    logging.getLogger(__name__).exception('Error during %s', context, exc_info=exc)
