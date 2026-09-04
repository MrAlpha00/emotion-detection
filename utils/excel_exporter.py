# =============================================================================
# Excel Report Generator
# =============================================================================
# Generates professional Excel reports from the user's database records.
# Uses openpyxl with formatting: bold headers, auto-widths, filters,
# frozen panes, borders, and proper number/date formatting.
# Never exports another user's data. Never generates placeholder data.
# =============================================================================

from io import BytesIO
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
from openpyxl.utils import get_column_letter
from config import Config
from models.detection import Detection
from models.live_session import LiveSession


def generate_excel_report(user):
    """
    Generate an Excel report for the given user's detection data.

    Args:
        user: The User model instance whose data to export.

    Returns:
        BytesIO buffer containing the Excel file, or None if no data.
    """
    # Query the user's data from the database
    detections = (
        Detection.query
        .filter_by(user_id=user.id)
        .order_by(Detection.detected_at.desc())
        .all()
    )

    live_sessions = (
        LiveSession.query
        .filter_by(user_id=user.id)
        .order_by(LiveSession.created_at.desc())
        .all()
    )

    # Create workbook
    wb = Workbook()

    # --- Common styles ---
    header_font = Font(bold=True, size=11, color='FFFFFF')
    header_fill = PatternFill(start_color='2C3E50', end_color='2C3E50', fill_type='solid')
    header_alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    cell_alignment = Alignment(horizontal='left', vertical='center')
    center_alignment = Alignment(horizontal='center', vertical='center')
    thin_border = Border(
        left=Side(style='thin', color='D5D8DC'),
        right=Side(style='thin', color='D5D8DC'),
        top=Side(style='thin', color='D5D8DC'),
        bottom=Side(style='thin', color='D5D8DC')
    )

    # =========================================================================
    # Sheet 1 — Detection History
    # =========================================================================
    ws1 = wb.active
    ws1.title = 'Detection History'

    detection_headers = [
        'S.No', 'User', 'Date', 'Time', 'Detection Type',
        'Emotion', 'Confidence (%)', 'Face Count', 'Session ID',
        'Detection Timestamp'
    ]

    # Write headers
    for col_idx, header in enumerate(detection_headers, 1):
        cell = ws1.cell(row=1, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_alignment
        cell.border = thin_border

    # Write detection data from the database
    for row_idx, detection in enumerate(detections, 2):
        detected_at = detection.detected_at

        row_data = [
            row_idx - 1,  # S.No
            user.username,
            detected_at.strftime(Config.DATE_DISPLAY_FORMAT) if detected_at else '',
            detected_at.strftime(Config.TIME_DISPLAY_FORMAT) if detected_at else '',
            detection.detection_type.capitalize() if detection.detection_type else '',
            detection.emotion,
            round(detection.confidence, 2) if detection.confidence else 0,
            detection.face_count if detection.face_count else 1,
            detection.session_id or '',
            detected_at.strftime(Config.TIMESTAMP_DISPLAY_FORMAT) if detected_at else ''
        ]

        for col_idx, value in enumerate(row_data, 1):
            cell = ws1.cell(row=row_idx, column=col_idx, value=value)
            cell.border = thin_border
            if col_idx in (1, 7, 8):  # S.No, Confidence, Face Count
                cell.alignment = center_alignment
            else:
                cell.alignment = cell_alignment

    # =========================================================================
    # Sheet 2 — Live Sessions
    # =========================================================================
    ws2 = wb.create_sheet(title='Live Sessions')

    session_headers = [
        'S.No', 'User', 'Start Date', 'Start Time',
        'End Date', 'End Time', 'Duration (seconds)',
        'Dominant Emotion', 'Average Confidence (%)', 'Total Detections'
    ]

    # Write headers
    for col_idx, header in enumerate(session_headers, 1):
        cell = ws2.cell(row=1, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_alignment
        cell.border = thin_border

    # Write live session data from the database
    for row_idx, sess in enumerate(live_sessions, 2):
        started = sess.started_at
        ended = sess.ended_at

        row_data = [
            row_idx - 1,  # S.No
            user.username,
            started.strftime(Config.DATE_DISPLAY_FORMAT) if started else '',
            started.strftime(Config.TIME_DISPLAY_FORMAT) if started else '',
            ended.strftime(Config.DATE_DISPLAY_FORMAT) if ended else '',
            ended.strftime(Config.TIME_DISPLAY_FORMAT) if ended else '',
            sess.duration_seconds if sess.duration_seconds else 0,
            sess.dominant_emotion or '',
            round(sess.average_confidence, 2) if sess.average_confidence else 0,
            sess.total_detections if sess.total_detections else 0
        ]

        for col_idx, value in enumerate(row_data, 1):
            cell = ws2.cell(row=row_idx, column=col_idx, value=value)
            cell.border = thin_border
            if col_idx in (1, 7, 9, 10):
                cell.alignment = center_alignment
            else:
                cell.alignment = cell_alignment

    # =========================================================================
    # Formatting for both sheets
    # =========================================================================
    for ws in [ws1, ws2]:
        # Auto-adjust column widths
        for col_idx in range(1, ws.max_column + 1):
            max_width = 0
            col_letter = get_column_letter(col_idx)
            for row in ws.iter_rows(min_col=col_idx, max_col=col_idx):
                for cell in row:
                    if cell.value:
                        max_width = max(max_width, len(str(cell.value)))
            ws.column_dimensions[col_letter].width = min(max_width + 4, 35)

        # Freeze the header row
        ws.freeze_panes = 'A2'

        # Add auto-filters
        if ws.max_row > 1:
            ws.auto_filter.ref = f'A1:{get_column_letter(ws.max_column)}{ws.max_row}'

    # Save to buffer
    buffer = BytesIO()
    wb.save(buffer)
    buffer.seek(0)

    return buffer
