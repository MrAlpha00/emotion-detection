# =============================================================================
# Dashboard Routes
# =============================================================================
# Handles the home/dashboard page, user profile, and conclusion page.
# All routes require authentication.
# =============================================================================

from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
from models.detection import Detection
from models.live_session import LiveSession
from utils.database import db
from sqlalchemy import func

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route('/')
@dashboard_bp.route('/home')
@login_required
def home():
    """Dashboard home page with user stats and quick actions."""

    # Gather user statistics for the dashboard
    total_detections = Detection.query.filter_by(user_id=current_user.id).count()
    total_sessions = LiveSession.query.filter_by(user_id=current_user.id).count()

    # Find the most common emotion
    most_common = (
        db.session.query(Detection.emotion, func.count(Detection.emotion).label('count'))
        .filter_by(user_id=current_user.id)
        .group_by(Detection.emotion)
        .order_by(func.count(Detection.emotion).desc())
        .first()
    )
    most_common_emotion = most_common[0] if most_common else None

    # Get the 5 most recent detections
    recent_detections = (
        Detection.query
        .filter_by(user_id=current_user.id)
        .order_by(Detection.detected_at.desc())
        .limit(5)
        .all()
    )

    return render_template('home.html',
                           total_detections=total_detections,
                           total_sessions=total_sessions,
                           most_common_emotion=most_common_emotion,
                           recent_detections=recent_detections)


@dashboard_bp.route('/profile')
@login_required
def profile():
    """User profile page with detection history."""

    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = 10

    # Get user's detection history, newest first
    detections = (
        Detection.query
        .filter_by(user_id=current_user.id)
        .order_by(Detection.detected_at.desc())
        .paginate(page=page, per_page=per_page, error_out=False)
    )

    # Get user's live sessions
    live_sessions = (
        LiveSession.query
        .filter_by(user_id=current_user.id)
        .order_by(LiveSession.created_at.desc())
        .all()
    )

    return render_template('user_module.html',
                           detections=detections,
                           live_sessions=live_sessions)


@dashboard_bp.route('/conclusion')
@login_required
def conclusion():
    """Academic conclusion page explaining the project."""
    return render_template('conclusion.html')
