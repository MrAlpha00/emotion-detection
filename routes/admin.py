# =============================================================================
# Admin Routes
# =============================================================================
# Dashboard statistics, user management, activity audit, analytics, detection
# browsing and Excel export.
#
# Every route below is wrapped in @admin_required, which enforces on the server
# that the caller is authenticated, that the account is active, and that the
# role is exactly 'admin'. Hiding links in the navbar is a UX nicety, not a
# security control.
#
# All statistics are computed from real database queries. Nothing on these pages
# is sampled, estimated or hard-coded.
# =============================================================================

import logging
from datetime import date, datetime, timedelta, timezone

from flask import (
    Blueprint,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_login import current_user
from sqlalchemy import func, or_
from sqlalchemy.exc import SQLAlchemyError

from config import Config
from models.detection import Detection
from models.live_session import LiveSession
from models.user import User
from models.user_activity import ActivityType, UserActivity
from utils.database import db
from utils.excel_exporter import build_admin_workbook
from utils.security import admin_required, log_activity
from utils.storage import get_storage, is_object_key_owned_by, is_valid_object_key

logger = logging.getLogger(__name__)

admin_bp = Blueprint('admin', __name__)

# Canonical emotion order used for every distribution report.
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Admin-generated Excel exports are bounded so a huge production database cannot
# exhaust serverless memory. The cap is reported in the UI when it is hit.
EXPORT_ROW_LIMIT = Config.MAX_PAGE_SIZE * 100


# =============================================================================
# Helpers
# =============================================================================
def _page_args():
    """Read and clamp pagination arguments from the query string."""
    page = max(request.args.get('page', 1, type=int) or 1, 1)
    per_page = request.args.get('per_page', Config.DEFAULT_PAGE_SIZE, type=int)
    if per_page is None:
        per_page = Config.DEFAULT_PAGE_SIZE
    per_page = max(10, min(per_page, Config.MAX_PAGE_SIZE))
    return page, per_page


def _parse_date(value):
    """Parse a YYYY-MM-DD query parameter, returning None when absent/invalid."""
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), '%Y-%m-%d').date()
    except (ValueError, TypeError):
        return None


def _day_start(day):
    """Timezone-aware UTC midnight for the given date."""
    return datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)


def _percent(part, whole):
    if not whole:
        return 0.0
    return round((part / whole) * 100.0, 1)


def _emotion_counts(query):
    """
    Count detections per emotion across an arbitrary query scope.

    Returns a list of ``{'emotion', 'count', 'percentage'}`` covering every
    canonical emotion, including the ones with a count of zero, so charts never
    silently drop a class.
    """
    rows = (
        query.with_entities(Detection.emotion, func.count(Detection.id))
        .group_by(Detection.emotion)
        .all()
    )
    counts = {emotion: 0 for emotion in EMOTIONS}
    for emotion, count in rows:
        if emotion in counts:
            counts[emotion] = count
        else:
            # An unexpected label (e.g. written before a config change) is still
            # reported rather than hidden.
            counts[emotion] = count

    total = sum(counts.values())
    return [
        {
            'emotion': emotion,
            'count': counts[emotion],
            'percentage': _percent(counts[emotion], total),
        }
        for emotion in counts
    ]


def _detections_by_day(query, days=14):
    """
    Detection counts for the last ``days`` days, zero-filled so a chart has no
    gaps. Uses a portable date truncation that works on SQLite and PostgreSQL.
    """
    start = _day_start(datetime.now(timezone.utc).date() - timedelta(days=days - 1))
    rows = (
        query.with_entities(func.date(Detection.detected_at), func.count(Detection.id))
        .filter(Detection.detected_at >= start)
        .group_by(func.date(Detection.detected_at))
        .all()
    )

    by_day = {}
    for raw_day, count in rows:
        # PostgreSQL returns a date object, SQLite a string.
        if isinstance(raw_day, str):
            try:
                raw_day = datetime.strptime(raw_day[:10], '%Y-%m-%d').date()
            except ValueError:
                continue
        if raw_day is not None:
            by_day[raw_day] = count

    result = []
    for offset in range(days):
        day = start.date() + timedelta(days=offset)
        result.append({'date': day.isoformat(), 'count': by_day.get(day, 0)})
    return result


def _parse_detection_filters(query):
    """Apply the detection table filters from the query string."""
    user_id = request.args.get('user_id', type=int)
    emotion = (request.args.get('emotion') or '').strip()
    detection_type = (request.args.get('detection_type') or '').strip()
    date_from = _parse_date(request.args.get('date_from'))
    date_to = _parse_date(request.args.get('date_to'))

    if user_id:
        query = query.filter(Detection.user_id == user_id)
    if emotion in EMOTIONS:
        query = query.filter(Detection.emotion == emotion)
    if detection_type in ('image', 'camera'):
        query = query.filter(Detection.detection_type == detection_type)
    if date_from:
        query = query.filter(Detection.detected_at >= _day_start(date_from))
    if date_to:
        # Inclusive of the whole end day.
        query = query.filter(Detection.detected_at < _day_start(date_to + timedelta(days=1)))

    return query


# =============================================================================
# Dashboard
# =============================================================================
@admin_bp.route('/admin')
@admin_bp.route('/admin/dashboard')
@admin_required
def dashboard():
    """Headline statistics, all derived from database aggregates."""
    today = datetime.now(timezone.utc).date()

    total_users = db.session.query(func.count(User.id)).scalar() or 0
    active_users = db.session.query(func.count(User.id)).filter(User.is_active.is_(True)).scalar() or 0
    inactive_users = total_users - active_users
    admin_users = db.session.query(func.count(User.id)).filter(User.role == Config.ROLE_ADMIN).scalar() or 0

    total_detections = db.session.query(func.count(Detection.id)).scalar() or 0
    image_detections = db.session.query(func.count(Detection.id)).filter(
        Detection.detection_type == 'image'
    ).scalar() or 0
    live_detections = total_detections - image_detections

    total_live_sessions = db.session.query(func.count(LiveSession.id)).scalar() or 0
    total_live_duration = db.session.query(func.coalesce(func.sum(LiveSession.duration_seconds), 0)).scalar() or 0

    todays_detections = db.session.query(func.count(Detection.id)).filter(
        func.date(Detection.detected_at) == today
    ).scalar() or 0

    average_confidence = db.session.query(func.coalesce(func.avg(Detection.confidence), 0)).scalar() or 0

    # Emotion distribution
    emotion_rows = (
        db.session.query(Detection.emotion, func.count(Detection.id))
        .group_by(Detection.emotion)
        .all()
    )
    emotion_map = {emotion: count for emotion, count in emotion_rows}
    emotion_counts = [
        {
            'emotion': emotion,
            'count': emotion_map.get(emotion, 0),
            'percentage': _percent(emotion_map.get(emotion, 0), total_detections),
        }
        for emotion in EMOTIONS
    ]

    # Per-user aggregates, one row per user (avoids N+1 queries on the user list)
    detection_totals = dict(
        db.session.query(Detection.user_id, func.count(Detection.id))
        .group_by(Detection.user_id)
        .all()
    )
    session_totals = dict(
        db.session.query(LiveSession.user_id, func.count(LiveSession.id))
        .group_by(LiveSession.user_id)
        .all()
    )
    duration_totals = dict(
        db.session.query(LiveSession.user_id, func.coalesce(func.sum(LiveSession.duration_seconds), 0))
        .group_by(LiveSession.user_id)
        .all()
    )

    top_users = (
        db.session.query(
            User.id,
            User.username,
            func.count(Detection.id).label('detection_count'),
        )
        .outerjoin(Detection, Detection.user_id == User.id)
        .group_by(User.id, User.username)
        .order_by(func.count(Detection.id).desc(), User.username.asc())
        .limit(10)
        .all()
    )

    recent_detections = (
        Detection.query
        .join(User, Detection.user_id == User.id)
        .with_entities(
            Detection.id,
            Detection.emotion,
            Detection.confidence,
            Detection.detection_type,
            Detection.detected_at,
            User.username,
        )
        .order_by(Detection.detected_at.desc())
        .limit(10)
        .all()
    )

    recent_activity = (
        UserActivity.query
        .order_by(UserActivity.created_at.desc())
        .limit(10)
        .all()
    )

    return render_template(
        'admin/dashboard.html',
        total_users=total_users,
        active_users=active_users,
        inactive_users=inactive_users,
        admin_users=admin_users,
        total_detections=total_detections,
        image_detections=image_detections,
        live_detections=live_detections,
        total_live_sessions=total_live_sessions,
        total_live_duration=total_live_duration,
        total_live_minutes=round((total_live_duration or 0) / 60.0, 1),
        todays_detections=todays_detections,
        average_confidence=round(average_confidence, 2),
        emotion_counts=emotion_counts,
        emotion_chart_labels=[row['emotion'] for row in emotion_counts],
        emotion_chart_data=[row['count'] for row in emotion_counts],
        detections_by_day=_detections_by_day(db.session.query(Detection)),
        top_users=top_users,
        recent_detections=recent_detections,
        recent_activity=recent_activity,
        detection_totals=detection_totals,
        session_totals=session_totals,
        duration_totals=duration_totals,
    )


# =============================================================================
# User management
# =============================================================================
@admin_bp.route('/admin/users')
@admin_required
def users():
    """Paginated, filterable user list with per-user usage metrics."""
    page, per_page = _page_args()

    search = (request.args.get('search') or '').strip()
    status = (request.args.get('status') or '').strip()
    role = (request.args.get('role') or '').strip()

    query = User.query

    if search:
        # SQLAlchemy parameterises the bound value; no string interpolation of
        # user input into SQL.
        pattern = f'%{search}%'
        query = query.filter(or_(User.username.ilike(pattern), User.email.ilike(pattern)))

    if status == 'active':
        query = query.filter(User.is_active.is_(True))
    elif status == 'inactive':
        query = query.filter(User.is_active.is_(False))

    if role in Config.ALLOWED_ROLES:
        query = query.filter(User.role == role)

    pagination = query.order_by(User.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    # Aggregates for the whole matching set, computed in three grouped queries
    # rather than one query per user.
    user_ids = [row.id for row in pagination.items]
    detection_counts = {}
    session_counts = {}
    duration_totals = {}
    last_detections = {}

    if user_ids:
        detection_counts = dict(
            db.session.query(Detection.user_id, func.count(Detection.id))
            .filter(Detection.user_id.in_(user_ids))
            .group_by(Detection.user_id)
            .all()
        )
        session_counts = dict(
            db.session.query(LiveSession.user_id, func.count(LiveSession.id))
            .filter(LiveSession.user_id.in_(user_ids))
            .group_by(LiveSession.user_id)
            .all()
        )
        duration_totals = dict(
            db.session.query(
                LiveSession.user_id, func.coalesce(func.sum(LiveSession.duration_seconds), 0)
            )
            .filter(LiveSession.user_id.in_(user_ids))
            .group_by(LiveSession.user_id)
            .all()
        )
        for user_id, detected_at in (
            db.session.query(
                func.max(Detection.detected_at).label('last'), Detection.user_id
            )
            .filter(Detection.user_id.in_(user_ids))
            .group_by(Detection.user_id)
            .all()
        ):
            last_detections[user_id] = detected_at

    return render_template(
        'admin/users.html',
        users=pagination,
        search=search,
        status=status,
        role_filter=role,
        detection_counts=detection_counts,
        session_counts=session_counts,
        duration_totals=duration_totals,
        last_detections=last_detections,
    )


@admin_bp.route('/admin/users/<int:user_id>')
@admin_required
def user_detail(user_id):
    """Full profile, usage summary and history for one user."""
    user = db.session.get(User, user_id)
    if user is None:
        flash('User not found.', 'warning')
        return redirect(url_for('admin.users'))

    detection_count = db.session.query(func.count(Detection.id)).filter(
        Detection.user_id == user.id
    ).scalar() or 0
    image_detections = db.session.query(func.count(Detection.id)).filter(
        Detection.user_id == user.id, Detection.detection_type == 'image'
    ).scalar() or 0
    live_detections = detection_count - image_detections

    live_session_count = db.session.query(func.count(LiveSession.id)).filter(
        LiveSession.user_id == user.id
    ).scalar() or 0
    total_live_duration = db.session.query(
        func.coalesce(func.sum(LiveSession.duration_seconds), 0)
    ).filter(LiveSession.user_id == user.id).scalar() or 0

    last_detection = (
        Detection.query.filter_by(user_id=user.id)
        .order_by(Detection.detected_at.desc())
        .first()
    )
    last_activity = (
        UserActivity.query.filter_by(user_id=user.id)
        .order_by(UserActivity.created_at.desc())
        .first()
    )

    page, per_page = _page_args()

    detections = (
        Detection.query.filter_by(user_id=user.id)
        .order_by(Detection.detected_at.desc())
        .paginate(page=page, per_page=min(per_page, 50), error_out=False)
    )
    live_sessions = (
        LiveSession.query.filter_by(user_id=user.id)
        .order_by(LiveSession.started_at.desc())
        .limit(20)
        .all()
    )
    activities = (
        UserActivity.query.filter_by(user_id=user.id)
        .order_by(UserActivity.created_at.desc())
        .limit(25)
        .all()
    )

    # Overall emotion profile for this single user.
    emotion_rows = (
        db.session.query(Detection.emotion, func.count(Detection.id))
        .filter(Detection.user_id == user.id)
        .group_by(Detection.emotion)
        .all()
    )
    emotion_map = dict(emotion_rows)
    emotion_profile = [
        {
            'emotion': emotion,
            'count': emotion_map.get(emotion, 0),
            'percentage': _percent(emotion_map.get(emotion, 0), detection_count),
        }
        for emotion in EMOTIONS
    ]

    return render_template(
        'admin/user_detail.html',
        user=user,
        detections=detections,
        live_sessions=live_sessions,
        activities=activities,
        detection_count=detection_count,
        image_detections=image_detections,
        live_detections=live_detections,
        live_session_count=live_session_count,
        total_live_duration=total_live_duration,
        total_live_minutes=round((total_live_duration or 0) / 60.0, 1),
        last_detection=last_detection,
        last_activity=last_activity,
        emotion_profile=emotion_profile,
        emotion_chart_labels=[row['emotion'] for row in emotion_profile],
        emotion_chart_data=[row['count'] for row in emotion_profile],
    )


@admin_bp.route('/admin/users/<int:user_id>/toggle-active', methods=['POST'])
@admin_required
def toggle_user_active(user_id):
    """
    Deactivate or reactivate an account.

    Deactivation is preferred over deletion: the user's history stays intact and
    login plus every detection route is blocked while the account is inactive.
    The acting admin cannot deactivate their own account.
    """
    user = db.session.get(User, user_id)
    if user is None:
        flash('User not found.', 'warning')
        return redirect(url_for('admin.users'))

    if user.id == current_user.id:
        flash('You cannot deactivate your own account.', 'danger')
        return redirect(url_for('admin.user_detail', user_id=user.id))

    # Losing the last admin would lock everyone out of the admin area.
    if user.is_active and user.role == Config.ROLE_ADMIN and _active_admin_count() <= 1:
        flash('At least one active administrator must remain.', 'danger')
        return redirect(url_for('admin.user_detail', user_id=user.id))

    user.is_active = not user.is_active
    db.session.commit()

    if user.is_active:
        log_activity(ActivityType.ADMIN_ACTIVATE_USER, details=f'target_user_id={user.id}')
        flash(f'{user.username} has been reactivated.', 'success')
    else:
        log_activity(ActivityType.ADMIN_DEACTIVATE_USER, details=f'target_user_id={user.id}')
        flash(
            f'{user.username} has been deactivated. They can no longer log in or run detections.',
            'success',
        )

    return redirect(request.referrer or url_for('admin.users'))


@admin_bp.route('/admin/users/<int:user_id>/change-role', methods=['POST'])
@admin_required
def change_user_role(user_id):
    """Promote or demote a user. Never applied to the acting admin."""
    user = db.session.get(User, user_id)
    if user is None:
        flash('User not found.', 'warning')
        return redirect(url_for('admin.users'))

    new_role = (request.form.get('role') or '').strip()
    if new_role not in Config.ALLOWED_ROLES:
        flash('Invalid role.', 'danger')
        return redirect(url_for('admin.user_detail', user_id=user.id))

    if user.id == current_user.id:
        flash('You cannot change your own role.', 'danger')
        return redirect(url_for('admin.user_detail', user_id=user.id))

    if user.role == Config.ROLE_ADMIN and new_role != Config.ROLE_ADMIN and _active_admin_count() <= 1:
        flash('At least one active administrator must remain.', 'danger')
        return redirect(url_for('admin.user_detail', user_id=user.id))

    previous_role = user.role
    user.role = new_role
    db.session.commit()

    log_activity(
        ActivityType.ADMIN_CHANGE_ROLE,
        details=f'target_user_id={user.id}; {previous_role}->{new_role}',
    )
    flash(f'{user.username} is now {new_role}.', 'success')
    return redirect(url_for('admin.user_detail', user_id=user.id))


@admin_bp.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@admin_required
def delete_user(user_id):
    """
    Permanently delete a user and everything that belongs to them.

    Related detections, live sessions and activity rows are removed through
    SQLAlchemy's delete-orphan cascade, so no orphaned records are left behind.
    The confirmation token must match the target id, which prevents an
    accidental or replayed form submission from deleting the wrong account.
    """
    user = db.session.get(User, user_id)
    if user is None:
        flash('User not found.', 'warning')
        return redirect(url_for('admin.users'))

    if user.id == current_user.id:
        flash('You cannot delete your own account.', 'danger')
        return redirect(url_for('admin.user_detail', user_id=user.id))

    if user.role == Config.ROLE_ADMIN and _active_admin_count() <= 1:
        flash('At least one active administrator must remain.', 'danger')
        return redirect(url_for('admin.user_detail', user_id=user.id))

    # Require an explicit confirmation that names the target account.
    typed = (request.form.get('confirm_username') or '').strip()
    if typed != user.username:
        flash(
            'Deletion cancelled: the confirmation did not match the username.',
            'danger',
        )
        return redirect(url_for('admin.user_detail', user_id=user.id))

    username = user.username
    target_id = user.id

    # Best-effort removal of the user's stored images. Ownership is re-checked
    # on every key so one user's delete can never touch another's files.
    try:
        storage = get_storage()
        keys = set()
        for detection in Detection.query.filter_by(user_id=target_id).all():
            for key in (detection.image_path, detection.processed_image_path):
                if key and is_valid_object_key(key) and is_object_key_owned_by(key, target_id):
                    keys.add(key)
        for key in keys:
            storage.delete(key)
    except Exception as exc:
        # A storage failure must not block the account deletion; log it and
        # carry on, since orphaned objects are harmless without a database row.
        logger.warning('Could not remove stored images for user %s: %s', target_id, exc)

    db.session.delete(user)
    db.session.commit()

    log_activity(ActivityType.ADMIN_DELETE_USER, details=f'target_user_id={target_id}')
    flash(f'User {username} and all of their records have been deleted.', 'success')
    return redirect(url_for('admin.users'))


def _active_admin_count():
    return db.session.query(func.count(User.id)).filter(
        User.role == Config.ROLE_ADMIN, User.is_active.is_(True)
    ).scalar() or 0


# =============================================================================
# Activity log
# =============================================================================
@admin_bp.route('/admin/activity')
@admin_required
def activity():
    """Paginated audit trail with user / action / date filters."""
    page, per_page = _page_args()

    query = UserActivity.query

    user_id = request.args.get('user_id', type=int)
    activity_type = (request.args.get('activity_type') or '').strip()
    date_from = _parse_date(request.args.get('date_from'))
    date_to = _parse_date(request.args.get('date_to'))

    if user_id:
        query = query.filter(UserActivity.user_id == user_id)
    if activity_type:
        query = query.filter(UserActivity.activity_type == activity_type)
    if date_from:
        query = query.filter(UserActivity.created_at >= _day_start(date_from))
    if date_to:
        query = query.filter(UserActivity.created_at < _day_start(date_to + timedelta(days=1)))

    pagination = query.order_by(UserActivity.created_at.desc(), UserActivity.id.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    # Filter dropdown options.
    users = User.query.order_by(User.username.asc()).all()
    used_types = [
        row[0]
        for row in db.session.query(UserActivity.activity_type)
        .distinct()
        .order_by(UserActivity.activity_type.asc())
        .all()
    ]
    activity_types = sorted(set(list(ActivityType.ALL) + used_types))

    return render_template(
        'admin/activity.html',
        activities=pagination,
        users=users,
        activity_types=activity_types,
        selected_user_id=user_id,
        selected_activity_type=activity_type,
        date_from=date_from.isoformat() if date_from else '',
        date_to=date_to.isoformat() if date_to else '',
    )


# =============================================================================
# Analytics
# =============================================================================
@admin_bp.route('/admin/analytics')
@admin_required
def analytics():
    """Database-backed detection analytics with Chart.js visualisations."""
    scope = (request.args.get('scope') or 'all').strip()
    days = request.args.get('days', 30, type=int)
    try:
        days = max(7, min(days or 30, 365))
    except (TypeError, ValueError):
        days = 30

    start = _day_start(datetime.now(timezone.utc).date() - timedelta(days=days - 1))

    base = Detection.query
    if scope == 'image':
        base = base.filter(Detection.detection_type == 'image')
    elif scope == 'live':
        base = base.filter(Detection.detection_type == 'camera')

    windowed = base.filter(Detection.detected_at >= start)

    # 1-2. Totals and emotion distribution
    total_detections = db.session.query(func.count(Detection.id)).scalar() or 0
    window_detections = windowed.with_entities(func.count(Detection.id)).scalar() or 0
    emotion_counts = _emotion_counts(windowed)

    # 3. Image vs live split
    image_detections = db.session.query(func.count(Detection.id)).filter(
        Detection.detection_type == 'image'
    ).scalar() or 0
    live_detections = total_detections - image_detections

    # 4. Detections by date (zero-filled)
    detections_by_day = _detections_by_day(windowed, days=days)

    # 5. Top users by detection count
    top_users = (
        db.session.query(User.username, func.count(Detection.id).label('count'))
        .join(Detection, Detection.user_id == User.id)
        .group_by(User.id, User.username)
        .order_by(func.count(Detection.id).desc())
        .limit(10)
        .all()
    )

    # 6-7. Live sessions by date and total duration
    total_live_sessions = db.session.query(func.count(LiveSession.id)).scalar() or 0
    total_live_duration = db.session.query(
        func.coalesce(func.sum(LiveSession.duration_seconds), 0)
    ).scalar() or 0

    session_rows = (
        db.session.query(func.date(LiveSession.started_at), func.count(LiveSession.id))
        .filter(LiveSession.started_at >= start)
        .group_by(func.date(LiveSession.started_at))
        .all()
    )
    sessions_by_day_map = {}
    for raw_day, count in session_rows:
        if isinstance(raw_day, str):
            try:
                raw_day = datetime.strptime(raw_day[:10], '%Y-%m-%d').date()
            except ValueError:
                continue
        if raw_day is not None:
            sessions_by_day_map[raw_day] = count

    sessions_by_day = []
    for offset in range(days):
        day = start.date() + timedelta(days=offset)
        sessions_by_day.append({
            'date': day.isoformat(),
            'count': sessions_by_day_map.get(day, 0),
        })

    # 8. Average confidence
    average_confidence = db.session.query(func.coalesce(func.avg(Detection.confidence), 0)).scalar() or 0

    # 9. Most common emotion
    top_emotion_row = (
        db.session.query(Detection.emotion, func.count(Detection.id))
        .group_by(Detection.emotion)
        .order_by(func.count(Detection.id).desc())
        .first()
    )
    most_common_emotion = top_emotion_row[0] if top_emotion_row else None
    most_common_count = top_emotion_row[1] if top_emotion_row else 0

    # 10. Recent activity
    recent_detections = (
        db.session.query(
            Detection.id,
            Detection.user_id,
            Detection.emotion,
            Detection.confidence,
            Detection.detection_type,
            Detection.detected_at,
            Detection.face_count,
            Detection.processing_time,
            User.username,
        )
        .join(User, Detection.user_id == User.id)
        .order_by(Detection.detected_at.desc())
        .limit(15)
        .all()
    )

    dominant_emotions = (
        db.session.query(LiveSession.dominant_emotion, func.count(LiveSession.id))
        .filter(LiveSession.dominant_emotion.isnot(None))
        .group_by(LiveSession.dominant_emotion)
        .order_by(func.count(LiveSession.id).desc())
        .limit(7)
        .all()
    )

    return render_template(
        'admin/analytics.html',
        total_detections=total_detections,
        window_detections=window_detections,
        image_detections=image_detections,
        live_detections=live_detections,
        total_live_sessions=total_live_sessions,
        total_live_duration=total_live_duration,
        total_live_minutes=round((total_live_duration or 0) / 60.0, 1),
        average_confidence=round(average_confidence, 2),
        emotion_counts=emotion_counts,
        detections_by_day=detections_by_day,
        sessions_by_day=sessions_by_day,
        top_users=top_users,
        most_common_emotion=most_common_emotion,
        most_common_count=most_common_count,
        dominant_emotions=dominant_emotions,
        recent_detections=recent_detections,
        days=days,
        scope=scope,
    )


# =============================================================================
# Detection browsing
# =============================================================================
@admin_bp.route('/admin/detections')
@admin_required
def detections():
    """Every detection in the system, with filters and pagination."""
    page, per_page = _page_args()

    query = _parse_detection_filters(Detection.query)
    pagination = query.order_by(Detection.detected_at.desc(), Detection.id.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    users = User.query.order_by(User.username.asc()).all()

    return render_template(
        'admin/detections.html',
        detections=pagination,
        users=users,
        emotions=EMOTIONS,
        selected_user_id=request.args.get('user_id', type=int),
        selected_emotion=request.args.get('emotion', ''),
        selected_detection_type=request.args.get('detection_type', ''),
        date_from=request.args.get('date_from', ''),
        date_to=request.args.get('date_to', ''),
    )


@admin_bp.route('/admin/detections/<int:detection_id>')
@admin_required
def detection_detail(detection_id):
    """Single detection with an authorized image preview."""
    detection = db.session.get(Detection, detection_id)
    if detection is None:
        flash('Detection not found.', 'warning')
        return redirect(url_for('admin.detections'))

    image_url = None
    if is_valid_object_key(detection.processed_image_path or detection.image_path or ''):
        image_url = url_for('admin.detection_image', detection_id=detection.id)

    return render_template(
        'admin/detection_detail.html',
        detection=detection,
        image_url=image_url,
    )


@admin_bp.route('/admin/detections/<int:detection_id>/image')
@admin_required
def detection_image(detection_id):
    """
    Stream a detection image for an authorised admin.

    Storage objects are never exposed as public URLs: this route checks the
    caller is an admin, then reads the object and streams it. For Supabase the
    download happens server-side using the secret key; the browser never sees
    the bucket URL.
    """
    detection = db.session.get(Detection, detection_id)
    if detection is None:
        return '', 404

    key = detection.processed_image_path or detection.image_path
    if not is_valid_object_key(key or ''):
        return '', 404

    storage = get_storage()
    payload = storage.load(key)
    if not payload:
        return '', 404

    data, content_type = payload
    return send_file(
        bytes(data),
        mimetype=content_type,
        max_age=0,
        conditional=False,
        download_name=f'{detection_id}.jpg',
    )


@admin_bp.route('/admin/detections/<int:detection_id>/delete', methods=['POST'])
@admin_required
def delete_detection(detection_id):
    """Delete a detection record and its stored images (ownership re-checked)."""
    detection = db.session.get(Detection, detection_id)
    if detection is None:
        flash('Detection not found.', 'warning')
        return redirect(url_for('admin.detections'))

    owner_id = detection.user_id
    try:
        storage = get_storage()
        for key in (detection.image_path, detection.processed_image_path):
            # is_object_key_owned_by guarantees we never delete another user's
            # object even if the database row were tampered with.
            if key and is_valid_object_key(key) and is_object_key_owned_by(key, owner_id):
                storage.delete(key)
    except Exception as exc:
        logger.warning('Could not delete stored image for detection %s: %s', detection_id, exc)

    db.session.delete(detection)
    db.session.commit()
    flash('Detection deleted.', 'success')
    return redirect(request.referrer or url_for('admin.detections'))


# =============================================================================
# Live sessions
# =============================================================================
@admin_bp.route('/admin/sessions')
@admin_required
def sessions():
    """All live sessions with owner, duration and dominant emotion."""
    page, per_page = _page_args()

    query = LiveSession.query

    user_id = request.args.get('user_id', type=int)
    date_from = _parse_date(request.args.get('date_from'))
    date_to = _parse_date(request.args.get('date_to'))

    if user_id:
        query = query.filter(LiveSession.user_id == user_id)
    if date_from:
        query = query.filter(LiveSession.started_at >= _day_start(date_from))
    if date_to:
        query = query.filter(LiveSession.started_at < _day_start(date_to + timedelta(days=1)))

    pagination = query.order_by(LiveSession.started_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    return render_template(
        'admin/sessions.html',
        sessions=pagination,
        users=User.query.order_by(User.username.asc()).all(),
        selected_user_id=user_id,
        date_from=request.args.get('date_from', ''),
        date_to=request.args.get('date_to', ''),
    )


# =============================================================================
# Exports
# =============================================================================
@admin_bp.route('/admin/exports', methods=['GET', 'POST'])
@admin_required
def exports():
    """Admin export centre. POST streams the workbook straight to the browser."""
    if request.method == 'POST':
        return _stream_admin_export()

    counts = {
        'users': db.session.query(func.count(User.id)).scalar() or 0,
        'detections': db.session.query(func.count(Detection.id)).scalar() or 0,
        'sessions': db.session.query(func.count(LiveSession.id)).scalar() or 0,
        'activities': db.session.query(func.count(UserActivity.id)).scalar() or 0,
    }

    recent_exports = (
        UserActivity.query.filter_by(activity_type=ActivityType.ADMIN_EXPORT)
        .order_by(UserActivity.created_at.desc())
        .limit(10)
        .all()
    )

    return render_template(
        'admin/exports.html',
        counts=counts,
        recent_exports=recent_exports,
        row_limit=EXPORT_ROW_LIMIT,
    )


def _stream_admin_export():
    """Build and stream the four-sheet admin workbook from query results."""
    try:
        users = User.query.order_by(User.created_at.asc()).limit(EXPORT_ROW_LIMIT).all()
        detections = (
            Detection.query.order_by(Detection.detected_at.asc()).limit(EXPORT_ROW_LIMIT).all()
        )
        live_sessions = (
            LiveSession.query.order_by(LiveSession.started_at.asc()).limit(EXPORT_ROW_LIMIT).all()
        )
        activities = (
            UserActivity.query.order_by(UserActivity.created_at.asc()).limit(EXPORT_ROW_LIMIT).all()
        )

        total_rows = len(users) + len(detections) + len(live_sessions) + len(activities)
        truncated = (
            db.session.query(func.count(Detection.id)).scalar() or 0
        ) > EXPORT_ROW_LIMIT

        buffer = build_admin_workbook(users, detections, live_sessions, activities)
        if buffer is None:
            flash('There is no data to export.', 'info')
            return redirect(url_for('admin.exports'))

        # Report date is stamped in the display timezone, matching the
        # timestamps inside the workbook.
        from utils.timezones import display_timezone

        filename = (
            f'Emotion_Admin_Export_'
            f'{datetime.now(display_timezone()).strftime("%Y-%m-%d")}.xlsx'
        )

        log_activity(
            ActivityType.ADMIN_EXPORT,
            details=f'rows={total_rows}; truncated={truncated}',
        )

        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            max_age=0,
        )
    except SQLAlchemyError as exc:
        db.session.rollback()
        logger.exception('Admin export failed')
        flash('The export could not be generated. Please try again.', 'danger')
        return redirect(url_for('admin.exports'))
    except Exception as exc:
        db.session.rollback()
        logger.exception('Admin export failed')
        flash('The export could not be generated. Please try again.', 'danger')
        return redirect(url_for('admin.exports'))


# =============================================================================
# JSON stats endpoint (used by the dashboard auto-refresh)
# =============================================================================
@admin_bp.route('/admin/api/stats')
@admin_required
def api_stats():
    """Small JSON payload for live dashboard refresh. Contains counts only."""
    today = datetime.now(timezone.utc).date()
    return jsonify({
        'total_users': db.session.query(func.count(User.id)).scalar() or 0,
        'active_users': db.session.query(func.count(User.id)).filter(
            User.is_active.is_(True)
        ).scalar() or 0,
        'total_detections': db.session.query(func.count(Detection.id)).scalar() or 0,
        'total_live_sessions': db.session.query(func.count(LiveSession.id)).scalar() or 0,
        'todays_detections': db.session.query(func.count(Detection.id)).filter(
            func.date(Detection.detected_at) == today
        ).scalar() or 0,
        'generated_at': datetime.now(timezone.utc).isoformat(),
    })
