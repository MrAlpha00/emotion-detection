# =============================================================================
# Excel Export Routes
# =============================================================================
# Generates and downloads an Excel report of the user's detection history
# and live sessions. Uses openpyxl for professional formatting.
# Only exports the current user's data — never another user's.
# =============================================================================

from io import BytesIO
from datetime import datetime, timezone
from flask import Blueprint, send_file, flash, redirect, url_for
from flask_login import login_required, current_user
from utils.excel_exporter import generate_excel_report

export_bp = Blueprint('export', __name__)


@export_bp.route('/export/results')
@login_required
def export_results():
    """Generate and download an Excel report of the user's detection history."""

    try:
        # Generate the Excel file from the database
        excel_buffer = generate_excel_report(current_user)

        if excel_buffer is None:
            flash('No data to export.', 'info')
            return redirect(url_for('dashboard.profile'))

        # Create the filename with username and date
        date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        filename = f"Emotion_Detection_Report_{current_user.username}_{date_str}.xlsx"

        return send_file(
            excel_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        flash('Failed to generate the Excel report.', 'danger')
        print(f"[Export] Error: {e}")
        return redirect(url_for('dashboard.profile'))
