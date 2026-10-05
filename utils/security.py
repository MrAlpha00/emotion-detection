# =============================================================================
# Security Helpers
# =============================================================================
# Server-side authorization primitives and the activity audit logger.
#
# Every check here runs on the backend. The UI hides admin links, but hiding a
# link is not authorization - the decorator below is what actually protects a
# route.
# =============================================================================

import logging
from functools import wraps

from flask import abort, flash, redirect, request, url_for
from flask_login import current_user

from utils.database import db

logger = logging.getLogger(__name__)


def client_ip():
    """
    Best-effort client IP address.

    On Vercel the real address arrives in X-Forwarded-For. Only the first entry
    is taken. This value is stored in the activity log for abuse investigation -
    see the privacy note in README.md.
    """
    forwarded = request.headers.get('X-Forwarded-For', '')
    if forwarded:
        return forwarded.split(',')[0].strip()[:45]
    real_ip = request.headers.get('X-Real-IP', '')
    if real_ip:
        return real_ip.strip()[:45]
    return (request.remote_addr or '')[:45]


def client_user_agent():
    return (request.headers.get('User-Agent') or '')[:500]


def log_activity(activity_type, user=None, details=None, commit=True):
    """
    Record an audited action.

    Never raises: an audit-log failure must not break the user's request. The
    failure is logged server-side instead.

    Args:
        activity_type: one of :class:`models.user_activity.ActivityType`
        user: the acting user; defaults to ``current_user``
        details: short, non-sensitive context string
    """
    from models.user_activity import UserActivity

    try:
        actor = user if user is not None else (current_user if current_user.is_authenticated else None)
        activity = UserActivity(
            user_id=getattr(actor, 'id', None),
            activity_type=activity_type,
            details=(details or '')[:500] or None,
            ip_address=client_ip(),
            user_agent=client_user_agent(),
        )
        db.session.add(activity)
        if commit:
            db.session.commit()
        return activity
    except Exception as exc:  # pragma: no cover - defensive
        db.session.rollback()
        logger.warning('Could not write activity log (%s): %s', activity_type, exc)
        return None


def admin_required(view_func):
    """
    Require an authenticated, active user whose role is 'admin'.

    Order of checks:
      1. authenticated?        -> redirect to login
      2. account active?       -> redirect to login (account disabled)
      3. role == 'admin'?      -> 403 Forbidden

    A 403 is returned for non-admins rather than a silent redirect, so probing
    /admin with a normal account is unambiguous and gets logged by the
    framework.
    """

    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('Please log in to access the admin area.', 'warning')
            return redirect(url_for('auth.login', next=request.path))
        if not getattr(current_user, 'is_active', False):
            flash('Your account has been deactivated.', 'danger')
            return redirect(url_for('auth.login'))
        if current_user.role != 'admin':
            abort(403)
        return view_func(*args, **kwargs)

    return wrapper


def active_user_required(view_func):
    """
    Require an authenticated *and* active account.

    Deactivating a user therefore immediately blocks every detection and history
    route, even if their session cookie is still technically valid.
    """

    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('auth.login', next=request.path))
        if not getattr(current_user, 'is_active', False):
            flash('Your account has been deactivated.', 'danger')
            logout_and_revoke()
            return redirect(url_for('auth.login'))
        return view_func(*args, **kwargs)

    return wrapper


def logout_and_revoke():
    """
    Drop the current session.

    Flask-Login sessions are client-side signed cookies, so the strongest thing
    we can do on deactivation is invalidate this session immediately and record
    the event. Any other session the user holds remains valid until it expires,
    which is why the ``active_user_required`` check above re-validates on every
    request rather than trusting the cookie alone.
    """
    from flask_login import logout_user

    try:
        log_activity('ACCOUNT_DEACTIVATED')
    except Exception:
        pass
    logout_user()


def owns_object(owner_id):
    """
    Ownership assertion for user-scoped records.

    Admin accounts are allowed through; everyone else must match ``owner_id``.

    Returns ``True`` when access is permitted, otherwise aborts with 403 so the
    caller never has to remember to handle the failure.
    """
    if current_user.is_authenticated and getattr(current_user, 'role', None) == 'admin':
        return True
    if owner_id is not None and current_user.is_authenticated and int(owner_id) == int(current_user.id):
        return True
    abort(403)
