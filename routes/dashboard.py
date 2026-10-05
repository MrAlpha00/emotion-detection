# =============================================================================
# Dashboard Routes
# =============================================================================
# Home page, user profile/history and the conclusion page.
#
# Every query here is scoped to `current_user.id`, so a user can only ever see
# their own detections, sessions and statistics. There is no route parameter
# that could be edited to reach another account's data.
# =============================================================================

import logging
from datetime import datetime, timedelta, timezone

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user
from sqlalchemy import func

from config import Config
from models.detection import Detection
from models.live_session import LiveSession
from utils.security import active_user_required

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint('dashboard', __name__)

EMOTIONS = Config.EMOTION_LABELS


@dashboard_bp.route('/')
@dashboard_bp.route('/home')
@active_user_required
def home():
    """Personal dashboard with the signed-in user's own statistics."""
    user_id = current_user.id

    total_detections = db_count(Detection, user_id)
    total_sessions = db_count(LiveSession, user_id)
    image_detections = count_where(Detection, user_id, Detection.detection_type == 'image')
    live_detections = total_detections - image_detections

    total_live_duration = (
        LiveSession.query.with_entities(
            func.coalesce(func.sum(LiveSession.duration_seconds), 0)
        )
        .filter(LiveSession.user_id == user_id)
        .scalar()
        or 0
    )

    most_common = (
        db_session()
        .query(Detection.emotion, func.count(Detection.emotion))
        .filter(Detection.user_id == user_id)
        .group_by(Detection.emotion)
        .order_by(func.count(Detection.emotion).desc())
        .first()
    )

    emotion_rows = (
        db_session()
        .query(Detection.emotion, func.count(Detection.id))
        .filter(Detection.user_id == user_id)
        .group_by(Detection.emotion)
        .all()
    )
    emotion_map = dict(emotion_rows)
    emotion_breakdown = [
        {
            'emotion': emotion,
            'count': emotion_map.get(emotion, 0),
            'percentage': round(emotion_map.get(emotion, 0) / total_detections * 100, 1)
            if total_detections
            else 0.0,
        }
        for emotion in EMOTIONS
    ]

    recent_detections = (
        Detection.query.filter_by(user_id=user_id)
        .order_by(Detection.detected_at.desc())
        .limit(5)
        .all()
    )

    recent_sessions = (
        LiveSession.query.filter_by(user_id=user_id)
        .order_by(LiveSession.started_at.desc())
        .limit(5)
        .all()
    )

    # Detections over the last 14 days, zero-filled for a clean chart.
    start = datetime.combine(
        datetime.now(timezone.utc).date() - timedelta(days=13),
        datetime.min.time(),
        tzinfo=timezone.utc,
    )
    day_rows = (
        db_session()
        .query(func.date(Detection.detected_at), func.count(Detection.id))
        .filter(Detection.user_id == user_id, Detection.detected_at >= start)
        .group_by(func.date(Detection.detected_at))
        .all()
    )
    day_map = {}
    for raw_day, count in day_rows:
        if isinstance(raw_day, str):
            try:
                raw_day = datetime.strptime(raw_day[:10], '%Y-%m-%d').date()
            except ValueError:
                continue
        if raw_day:
            day_map[raw_day] = count

    activity_chart = []
    # `start` is already a UTC midnight datetime, so `start.date()` is a date and
    # adding a plain int is invalid; the offset must be a timedelta.
    first_day = start.date()
    for offset in range(14):
        day = first_day + timedelta(days=offset)
        activity_chart.append({
            'date': day.isoformat(),
            'count': day_map.get(day, 0),
        })

    return render_template(
        'home.html',
        total_detections=total_detections,
        total_sessions=total_sessions,
        image_detections=image_detections,
        live_detections=live_detections,
        total_live_duration=total_live_duration,
        total_live_minutes=round((total_live_duration or 0) / 60.0, 1),
        most_common_emotion=most_common[0] if most_common else None,
        emotion_breakdown=emotion_breakdown,
        emotion_chart_labels=[row['emotion'] for row in emotion_breakdown],
        emotion_chart_data=[row['count'] for row in emotion_breakdown],
        activity_chart=activity_chart,
        recent_detections=recent_detections,
        recent_sessions=recent_sessions,
    )


@dashboard_bp.route('/profile')
@active_user_required
def profile():
    """Personal history: paginated detections plus live session summaries."""
    page = max(request.args.get('page', 1, type=int) or 1, 1)
    per_page = max(5, min(request.args.get('per_page', 10, type=int) or 10, 50))

    emotion_filter = (request.args.get('emotion') or '').strip()
    type_filter = (request.args.get('type') or '').strip()

    query = Detection.query.filter_by(user_id=current_user.id)

    if emotion_filter in EMOTIONS:
        query = query.filter(Detection.emotion == emotion_filter)
    if type_filter in ('image', 'camera'):
        query = query.filter(Detection.detection_type == type_filter)

    detections = query.order_by(Detection.detected_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    live_sessions = (
        LiveSession.query.filter_by(user_id=current_user.id)
        .order_by(LiveSession.started_at.desc())
        .limit(25)
        .all()
    )

    return render_template(
        'user_module.html',
        detections=detections,
        live_sessions=live_sessions,
        emotions=EMOTIONS,
        emotion_filter=emotion_filter,
        type_filter=type_filter,
    )


@dashboard_bp.route('/conclusion')
@active_user_required
def conclusion():
    """Academic conclusion page."""
    return render_template('conclusion.html')


def db_session():
    """Shortcut to the SQLAlchemy session (kept local to avoid a wide import)."""
    from utils.database import db

    return db.session


def db_count(model, user_id):
    return model.query.filter_by(user_id=user_id).count()


def count_where(model, user_id, *conditions):
    return model.query.filter(model.user_id == user_id, *conditions).count()
