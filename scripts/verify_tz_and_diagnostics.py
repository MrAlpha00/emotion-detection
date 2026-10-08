"""Targeted verification for the timezone display fix + prediction diagnostics.

Run from the project root:  venv\\Scripts\\python.exe <this file>
"""
import logging
import pathlib
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

failures = []


def check(name, fn):
    try:
        fn()
        print(f'[PASS] {name}')
    except Exception as exc:  # noqa: BLE001
        failures.append(name)
        print(f'[FAIL] {name}: {exc!r}')


# ---------------------------------------------------------------------------
# 1. Timezone conversion (zone-aware, no hardcoded offsets)
# ---------------------------------------------------------------------------
def t_conversions():
    from utils.timezones import as_utc, format_display, to_display

    # Naive value straight from the DB (stored UTC) -> IST wall clock.
    naive_utc = datetime(2026, 1, 1, 10, 0, 0)
    d1 = to_display(naive_utc)
    assert d1.tzinfo is not None, 'display value must be timezone-aware'
    assert d1.utcoffset().total_seconds() == 5.5 * 3600, d1.utcoffset()
    assert (d1.hour, d1.minute) == (15, 30), d1

    # Aware UTC instant -> same conversion.
    aware = datetime(2026, 6, 1, 4, 0, tzinfo=timezone.utc)  # June: no IST DST
    d2 = to_display(aware)
    assert (d2.hour, d2.minute) == (9, 30), d2

    # ISO-8601 string (session payload on the result page) -> same instant.
    d3 = to_display(aware.isoformat())
    assert d3 == d2, (d3, d2)

    # Midnight UTC in IST rolls to the next calendar day.
    rollover = to_display(datetime(2026, 1, 1, 18, 30, tzinfo=timezone.utc))
    assert (rollover.day, rollover.hour) == (2, 0), rollover

    assert to_display(None) is None
    assert as_utc(naive_utc).tzinfo is not None
    assert format_display(naive_utc, '%Y-%m-%d %H:%M') == '2026-01-01 15:30'


# ---------------------------------------------------------------------------
# 2. Every template compiles with the |to_local filter registered
# ---------------------------------------------------------------------------
def t_templates_compile():
    from jinja2 import Environment, FileSystemLoader

    from utils.timezones import to_display

    root = pathlib.Path('templates')
    env = Environment(loader=FileSystemLoader(str(root)))
    env.filters['to_local'] = to_display
    for path in sorted(root.rglob('*.html')):
        env.get_template(str(path.relative_to(root)).replace('\\', '/'))

    # No template may still render a stored timestamp without the filter.
    stray = []
    import re

    pat = re.compile(
        r'\w+\.(?:detected_at|created_at|started_at|ended_at|last_login_at'
        r'|updated_at)\.strftime\('
    )
    for path in root.rglob('*.html'):
        if pat.search(path.read_text(encoding='utf-8')):
            stray.append(path.name)
    assert not stray, f'unconverted strftime calls in {stray}'


# ---------------------------------------------------------------------------
# 3. Prediction diagnostics: raw probabilities logged, result unchanged
# ---------------------------------------------------------------------------
def t_prediction_diagnostics():
    import numpy as np

    from config import Config
    from utils.emotion_predictor import EmotionPredictor, diagnostic_logger

    class StubModel:
        input_shape = (
            None,
            Config.MODEL_INPUT_SIZE[0],
            Config.MODEL_INPUT_SIZE[1],
            1 if Config.MODEL_COLOR_MODE == 'grayscale' else 3,
        )
        output_shape = (None, Config.NUM_EMOTION_CLASSES)

        def predict(self, x, verbose=0):
            vec = np.zeros((1, Config.NUM_EMOTION_CLASSES), dtype='float32')
            labels = Config.EMOTION_LABELS
            vec[0, labels.index('Happy')] = 0.61
            vec[0, labels.index('Sad')] = 0.22
            vec[0, labels.index('Neutral')] = 0.17
            return vec

    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Capture()
    diagnostic_logger.addHandler(handler)
    diagnostic_logger.setLevel(logging.INFO)
    try:
        predictor = EmotionPredictor()
        predictor._model = StubModel()
        predictor._available = True
        predictor._status_message = 'stub'

        face = np.zeros((60, 60, 3), dtype='uint8')
        result = predictor.predict(face, face_context='image: largest of 2 face(s)')

        # Class mapping / argmax behaviour untouched.
        assert result['emotion'] == 'Happy', result['emotion']
        assert abs(result['confidence'] - 61.0) < 0.01, result['confidence']
        assert result['probabilities']['Sad'] == 22.0, result['probabilities']
        assert len(result['probabilities']) == Config.NUM_EMOTION_CLASSES

        # The full raw vector is logged for later quality evaluation.
        assert len(records) == 1, records
        line = records[0]
        assert 'PREDICT_PROBS' in line, line
        assert 'face=image: largest of 2 face(s)' in line, line
        assert 'predicted=Happy' in line and 'confidence=61.00%' in line, line
        assert 'sum=100.00%' in line, line
        for label in Config.EMOTION_LABELS:
            assert f'{label}:' in line, (label, line)
        print('      sample log line:', line)

        # Flag off -> nothing logged.
        records.clear()
        original_flag = Config.PREDICT_LOG_PROBABILITIES
        Config.PREDICT_LOG_PROBABILITIES = False
        try:
            predictor.predict(face, face_context='disabled')
        finally:
            Config.PREDICT_LOG_PROBABILITIES = original_flag
        assert records == [], records
    finally:
        diagnostic_logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# 4. Excel export uses the display timezone
# ---------------------------------------------------------------------------
def t_excel_ist():
    from openpyxl import load_workbook

    from utils.excel_exporter import build_user_workbook

    det = SimpleNamespace(
        id=1,
        user=None,
        detected_at=datetime(2026, 1, 1, 10, 0, 0),  # naive UTC from the DB
        detection_type='image',
        emotion='Happy',
        confidence=61.0,
        face_count=1,
        processing_time=0.6,
    )
    session = SimpleNamespace(
        id=7,
        user=None,
        started_at=datetime(2026, 1, 1, 18, 30, 0),   # 00:00 IST next day
        ended_at=datetime(2026, 1, 1, 18, 45, 0),
        duration_seconds=900,
        dominant_emotion='Happy',
        average_confidence=61.0,
        total_detections=3,
    )
    buf = build_user_workbook([det], [session], username='tester')
    wb = load_workbook(buf)

    d_headers = [c.value for c in wb['Detection History'][1]]
    d_row = [c.value for c in wb['Detection History'][2]]
    assert 'IST' in d_headers[-1], d_headers
    assert d_row[2] == '01 Jan 2026', d_row
    assert d_row[3] == '03:30:00 PM', d_row  # 10:00 UTC -> 15:30 IST
    assert d_row[9] == '01 January 2026, 03:30:00 PM', d_row

    s_row = [c.value for c in wb['Live Sessions'][2]]
    assert s_row[2] == '02 Jan 2026', s_row  # 18:30 UTC -> next day IST
    assert s_row[3] == '12:00:00 AM', s_row


# ---------------------------------------------------------------------------
# 5. Config wiring
# ---------------------------------------------------------------------------
def t_config():
    from config import Config

    assert Config.DISPLAY_TIMEZONE == 'Asia/Kolkata', Config.DISPLAY_TIMEZONE
    assert Config.DISPLAY_TIMEZONE_LABEL == 'IST'
    assert Config.PREDICT_LOG_PROBABILITIES is True


check('UTC stored values convert to IST via zoneinfo', t_conversions)
check('all templates compile and use |to_local', t_templates_compile)
check('PREDICT_PROBS logs raw vector; prediction unchanged', t_prediction_diagnostics)
check('Excel export renders IST timestamps and headers', t_excel_ist)
check('timezone config is wired', t_config)

print()
print('=' * 60)
if failures:
    print(f'  FAILURES: {failures}')
    sys.exit(1)
print('  ALL TARGETED CHECKS PASSED')
print('=' * 60)
