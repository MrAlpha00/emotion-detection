# =============================================================================
# Authentication Routes
# =============================================================================
# Handles user registration, login, and logout.
# Uses Flask-Login for session management and Werkzeug for password hashing.
# All form validation is done server-side.
# =============================================================================

import re
from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from models.user import User
from utils.database import db

auth_bp = Blueprint('auth', __name__)


def is_valid_email(email):
    """Basic email format validation."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def is_valid_username(username):
    """Username must be 3-80 chars, alphanumeric with underscores/hyphens."""
    pattern = r'^[a-zA-Z0-9_-]{3,80}$'
    return re.match(pattern, username) is not None


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Login page — authenticate with email/username and password."""
    # Already logged in users go to dashboard
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.home'))

    if request.method == 'POST':
        login_input = request.form.get('login_input', '').strip()
        password = request.form.get('password', '')

        if not login_input or not password:
            flash('Please enter your email/username and password.', 'danger')
            return render_template('login.html')

        # Allow login with either email or username
        user = User.query.filter(
            (User.email == login_input) | (User.username == login_input)
        ).first()

        if user and user.check_password(password):
            login_user(user)
            flash(f'Welcome back, {user.username}!', 'success')
            # Redirect to the page the user was trying to access, or dashboard
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard.home'))
        else:
            flash('Invalid email/username or password.', 'danger')

    return render_template('login.html')


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page — create a new user account."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.home'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # --- Validation ---
        errors = []

        if not username:
            errors.append('Username is required.')
        elif not is_valid_username(username):
            errors.append('Username must be 3-80 characters (letters, numbers, underscores, hyphens).')

        if not email:
            errors.append('Email is required.')
        elif not is_valid_email(email):
            errors.append('Please enter a valid email address.')

        if not password:
            errors.append('Password is required.')
        elif len(password) < 6:
            errors.append('Password must be at least 6 characters long.')

        if password != confirm_password:
            errors.append('Passwords do not match.')

        # Check for duplicate username/email
        if not errors:
            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                errors.append('This username is already taken.')

            existing_email = User.query.filter_by(email=email).first()
            if existing_email:
                errors.append('An account with this email already exists.')

        if errors:
            for error in errors:
                flash(error, 'danger')
            return render_template('register.html')

        # --- Create user ---
        try:
            new_user = User(username=username, email=email)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('auth.login'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred during registration. Please try again.', 'danger')
            print(f"[Auth] Registration error: {e}")

    return render_template('register.html')


@auth_bp.route('/logout')
@login_required
def logout():
    """Logout — clear the user session."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))
