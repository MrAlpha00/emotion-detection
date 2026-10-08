# =============================================================================
# Excel Export Routes (user scope)
# =============================================================================
# Generates the signed-in user's own history as an Excel workbook, in memory.
#
# Authorisation rule: there is deliberately NO user_id parameter. The export
# scope is always `current_user`, so a user cannot export somebody else's data
# by editing the URL. Admins use the separate, role-protected admin export.
# =============================================================================

import logging
from datetime import datetime

from flask import Blueprint, flash, redirect, send_file, url_for
from flask_login import current_user

from models.detection import Detection
from models.live_session import LiveSession
from models.user_activity import ActivityType
from utils.database import db
from utils.excel_exporter import build_user_workbook
from utils.security import active_user_required, log_activity

logger = logging.getLogger(__name__)

export_bp = Blueprint('export', __name__)

# Guard against pulling an unbounded number of rows into memory on a serverless
# instance with a limited heap.
EXPORT_ROW_LIMIT = 50000


@export_bp.route('/export/results')
@active_user_required
def export_results():
    """Download an Excel report of the current user's own detection history."""
    try:
        detections = (
            Detection.query.filter_by(user_id=current_user.id)
            .order_by(Detection.detected_at.desc())
            .limit(EXPORT_ROW_LIMIT)
            .all()
        )
        live_sessions = (
            LiveSession.query.filter_by(user_id=current_user.id)
            .order_by(LiveSession.started_at.desc())
            .limit(EXPORT_ROW_LIMIT)
            .all()
        )

        buffer = build_user_workbook(detections, live_sessions, username=current_user.username)

        if buffer is None:
            flash('There is no data to export yet.', 'info')
            return redirect(url_for('dashboard.profile'))

        # Report date is stamped in the user's display timezone, matching the
        # timestamps inside the workbook.
        from utils.timezones import display_timezone

        filename = (
            f'Emotion_Detection_Report_{current_user.username}_'
            f'{datetime.now(display_timezone()).strftime("%Y-%m-%d")}.xlsx'
        )

        log_activity(
            ActivityType.EXPORT,
            details=f'detections={len(detections)}; sessions={len(live_sessions)}',
        )

        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            max_age=0,
        )

    except Exception:
        db.session.rollback()
        logger.exception('User export failed')
        flash('The report could not be generated. Please try again.', 'danger')
        return redirect(url_for('dashboard.profile'))
