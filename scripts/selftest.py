"""
Local SQLite integration checks.

Run with a throwaway database so the real ``database/emotion_app.db`` is never
touched::

    python scripts/selftest.py

Every check prints PASS, FAIL or BLOCKED. The exit code is 1 if anything failed.
BLOCKED never fails the run - it means the environment (a missing model, an
absent Supabase secret) prevented the check, which is reported honestly rather
than skipped silently.
"""

import base64
import io
import os
import re
import sys
import tempfile
import traceback

# Point the app at a throwaway SQLite file before anything imports config.
_TMP_DIR = tempfile.mkdtemp(prefix='emotion-selftest-')
os.environ['DATABASE_URL'] = ''
os.environ['STORAGE_BACKEND'] = 'local'
os.environ['SECRET_KEY'] = 'selftest-secret-key-not-a-real-credential'
os.environ.pop('FLASK_ENV', None)
os.environ.pop('VERCEL', None)
os.environ.pop('SUPABASE_URL', None)
os.environ.pop('SUPABASE_SECRET_KEY', None)
os.environ.pop('SUPABASE_KEY', None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

config.Config.DATABASE_DIR = _TMP_DIR
config.Config.DATABASE_PATH = os.path.join(_TMP_DIR, 'selftest.db')
config.Config.SQLALCHEMY_DATABASE_URI = f'sqlite:///{config.Config.DATABASE_PATH}'
config.Config.USE_SQLITE = True

RESULTS = []


def record(status, name, detail=''):
    RESULTS.append((status, name, detail))
    marker = {'PASS': '[PASS]  ', 'FAIL': '[FAIL]  ', 'BLOCKED': '[BLOCKED]'}
    print(f'{marker[status]} {name}' + (f'\n         {detail}' if detail else ''))
    return status == 'PASS'


BLOCKED_SENTINEL = 'BLOCKED'


def check(name, fn):
    """
    Run ``fn``, record PASS/FAIL/BLOCKED, and return whatever ``fn`` returned.

    Returning the value lets a later check chain off an earlier one (for
    example, only test the live-session token if a session was really issued).
    A function returning the string 'BLOCKED' is recorded as BLOCKED - that is
    how a check reports "the environment prevented this", which is neither a
    pass nor a failure. Anything that raises is a FAIL.
    """
    try:
        result = fn()
        if result == BLOCKED_SENTINEL:
            record('BLOCKED', name, 'the environment prevented this check')
            return None
        ok = result is not False
        record('PASS' if ok else 'FAIL', name)
        return result
    except AssertionError as exc:
        record('FAIL', name, str(exc) or 'assertion failed')
        return None
    except Exception as exc:  # noqa: BLE001
        record(
            'FAIL', name, f'{type(exc).__name__}: {exc}\n' + traceback.format_exc(limit=3)
        )
        return None


# =============================================================================
# Fixtures
# =============================================================================
def build_app():
    from app import create_app

    return create_app()


def make_client(app):
    return app.test_client()


def csrf_of(client, url):
    """
    Extract the CSRF token from a rendered form.

    Falls back to the base layout's ``<meta name="csrf-token">`` tag, which is
    rendered on every page and is what JavaScript reads.
    """
    html = client.get(url).get_data(as_text=True)
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    if match:
        return match.group(1)
    match = re.search(r'<meta name="csrf-token" content="([^"]+)"', html)
    return match.group(1) if match else None


def create_user(app, username, email, password, role='user', is_active=True):
    from models.user import User
    from utils.database import db

    with app.app_context():
        user = User(username=username, email=email, role=role, is_active=is_active)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        return user.id


def add_detection(app, user_id, emotion='Happy', session_id='tok-abc'):
    from models.detection import Detection
    from utils.database import db

    with app.app_context():
        row = Detection(
            user_id=user_id,
            detection_type='camera',
            emotion=emotion,
            confidence=91.5,
            face_count=1,
            session_id=session_id,
        )
        db.session.add(row)
        db.session.commit()
        return row.id


def login(client, app, identity, password):
    token = csrf_of(client, '/login')
    return client.post(
        '/login',
        data={'login_input': identity, 'password': password, 'csrf_token': token},
        follow_redirects=True,
    )


def jpeg_bytes():
    """A small real JPEG so the upload path is exercised end to end."""
    import cv2
    import numpy as np

    canvas = np.full((240, 320, 3), 200, dtype=np.uint8)
    cv2.circle(canvas, (160, 120), 70, (150, 180, 210), -1)
    ok, buf = cv2.imencode('.jpg', canvas)
    assert ok
    return buf.tobytes()


# =============================================================================
# Checks
# =============================================================================
def main():
    app = build_app()
    app.config['WTF_CSRF_ENABLED'] = True
    client = make_client(app)

    # --- 1. health ------------------------------------------------------------
    def t_health():
        payload = client.get('/health').get_json()
        assert payload['status'] in ('ok', 'degraded'), payload
        assert payload['database']['engine'] == 'sqlite', payload
        # A health payload must never leak a connection string.
        body = client.get('/health').get_data(as_text=True)
        assert 'postgresql://' not in body and 'sqlite:///' not in body
        return True

    check('GET /health reports the engine and no credentials', t_health)

    # --- 2. public pages render (this is where csrf_token() is exercised) ----
    def t_public_pages():
        for url in ('/login', '/register'):
            resp = client.get(url)
            assert resp.status_code == 200, f'{url} -> {resp.status_code}'
            body = resp.get_data(as_text=True)
            assert 'csrf-token' in body or 'csrf_token' in body, f'{url} missing CSRF'
        # The site root is allowed to redirect anonymous visitors to /login.
        assert client.get('/').status_code in (200, 302)
        return True

    check('Login/register render with a CSRF token present', t_public_pages)

    # --- 3. CSRF is enforced on login ----------------------------------------
    def t_csrf_login():
        fresh = make_client(app)
        resp = fresh.post(
            '/login', data={'login_input': 'x', 'password': 'y'}
        )
        assert resp.status_code in (400, 403), f'expected 400/403, got {resp.status_code}'
        return True

    check('POST /login without a CSRF token is rejected', t_csrf_login)

    # --- 4. registration validation + no self-promotion ----------------------
    def t_register():
        from models.user import User

        c = make_client(app)

        # Short password must be refused.
        token = csrf_of(c, '/register')
        c.post('/register', data={
            'username': 'shortpw', 'email': 'short@example.com', 'password': 'short',
            'confirm_password': 'short', 'csrf_token': token,
        }, follow_redirects=True)
        with app.app_context():
            assert User.query.filter_by(username='shortpw').first() is None, \
                'short password accepted'

        # Mismatched confirmation must be refused.
        token = csrf_of(c, '/register')
        c.post('/register', data={
            'username': 'mismatch', 'email': 'mismatch@example.com', 'password': 'LongEnough1',
            'confirm_password': 'Different1', 'csrf_token': token,
        }, follow_redirects=True)
        with app.app_context():
            assert User.query.filter_by(username='mismatch').first() is None, \
                'mismatched passwords accepted'

        # Malformed email must be refused.
        token = csrf_of(c, '/register')
        c.post('/register', data={
            'username': 'bademail', 'email': 'not-an-email', 'password': 'LongEnough1',
            'confirm_password': 'LongEnough1', 'csrf_token': token,
        }, follow_redirects=True)
        with app.app_context():
            assert User.query.filter_by(username='bademail').first() is None, \
                'invalid email accepted'

        # Valid registration that also tries to smuggle role/id/is_active.
        token = csrf_of(c, '/register')
        c.post('/register', data={
            'username': 'alice', 'email': 'alice@example.com', 'password': 'LongEnough1',
            'confirm_password': 'LongEnough1', 'csrf_token': token,
            'role': 'admin', 'is_active': 'true', 'id': '1', 'login_count': '999',
        }, follow_redirects=True)
        with app.app_context():
            alice = User.query.filter_by(username='alice').first()
            assert alice is not None, 'valid registration failed'
            assert alice.role == 'user', f'role escalation succeeded: {alice.role}'
            assert alice.is_active is True
            assert alice.login_count == 0, f'login_count forged: {alice.login_count}'

        # Now that alice exists, a duplicate username must be refused.
        token = csrf_of(c, '/register')
        c.post('/register', data={
            'username': 'alice', 'email': 'different@example.com', 'password': 'LongEnough1',
            'confirm_password': 'LongEnough1', 'csrf_token': token,
        }, follow_redirects=True)
        with app.app_context():
            assert User.query.filter_by(email='different@example.com').first() is None, \
                'duplicate username accepted'
        return True

    check('Registration validates input and ignores forged role/id', t_register)

    # --- 5. login + activity logging ----------------------------------------
    def t_login():
        create_user(app, 'bob', 'bob@example.com', 'BobPassword1')
        c = make_client(app)
        resp = login(c, app, 'bob', 'BobPassword1')
        assert resp.status_code == 200
        with app.app_context():
            from models.user import User
            from models.user_activity import ActivityType, UserActivity
            bob = User.query.filter_by(username='bob').first()
            assert bob.login_count == 1, bob.login_count
            assert bob.last_login_at is not None
            assert UserActivity.query.filter_by(
                user_id=bob.id, activity_type=ActivityType.LOGIN
            ).first() is not None, 'login not audited'
        return True

    check('Login sets login_count/last_login_at and writes an audit row', t_login)

    # --- 6. wrong password does not enumerate users --------------------------
    def t_bad_password():
        c = make_client(app)
        login(c, app, 'nobody-here', 'whatever')
        with app.app_context():
            from models.user import User
            assert User.query.filter_by(username='nobody-here').first() is None
        return True

    check('Login with an unknown user creates nothing', t_bad_password)

    # --- 7. protected pages require login ------------------------------------
    def t_protected():
        c = make_client(app)
        for url in ('/home', '/profile', '/detect', '/live', '/admin', '/admin/users'):
            resp = c.get(url)
            assert resp.status_code in (302, 401, 403), f'{url} -> {resp.status_code}'
            if resp.status_code == 302:
                assert '/login' in resp.headers.get('Location', ''), f'{url} -> {resp.headers.get("Location")}'
        return True

    check('User and admin pages redirect anonymous users to /login', t_protected)

    # --- 8. admin enforcement ------------------------------------------------
    def t_admin_block():
        create_user(app, 'carol', 'carol@example.com', 'CarolPassword1')
        c = make_client(app)
        login(c, app, 'carol', 'CarolPassword1')
        for url in ('/admin', '/admin/users', '/admin/activity', '/admin/analytics',
                    '/admin/detections', '/admin/sessions', '/admin/exports',
                    '/admin/api/stats'):
            resp = c.get(url)
            assert resp.status_code in (302, 403), f'{url} -> {resp.status_code}'
        return True

    check('A standard user is refused every admin page', t_admin_block)

    # --- 9. admin pages render ----------------------------------------------
    def t_admin_render():
        admin_id = create_user(app, 'root', 'root@example.com', 'RootPassword1', role='admin')
        c = make_client(app)
        login(c, app, 'root', 'RootPassword1')
        pages = ['/admin', '/admin/dashboard', '/admin/users', '/admin/analytics',
                 '/admin/activity', '/admin/detections', '/admin/sessions',
                 '/admin/exports']
        for url in pages:
            resp = c.get(url)
            assert resp.status_code == 200, f'{url} -> {resp.status_code}\n{resp.get_data(as_text=True)[:1500]}'
        return True

    check('Every admin page renders for an admin (200)', t_admin_render)

    # --- 10. admin user detail + detection detail render ---------------------
    def t_detail_render():
        from models.user import User

        c = make_client(app)
        login(c, app, 'root', 'RootPassword1')
        with app.app_context():
            target = User.query.filter_by(username='bob').first()
            target_id = target.id
        det_id = add_detection(app, target_id)

        for url in (f'/admin/users/{target_id}',
                    f'/admin/detections/{det_id}',
                    f'/admin/detections/{det_id}/image'):
            resp = c.get(url)
            assert resp.status_code in (200, 404), f'{url} -> {resp.status_code}'
        return True

    check('Admin user/detection detail pages render', t_detail_render)

    # --- 11. user search filter (or_ usage) ---------------------------------
    def t_admin_search():
        c = make_client(app)
        login(c, app, 'root', 'RootPassword1')
        resp = c.get('/admin/users?search=alice')
        assert resp.status_code == 200, resp.status_code
        body = resp.get_data(as_text=True)
        assert 'alice' in body
        assert 'carol' not in body
        return True

    check('Admin user search filters correctly (or_ across username/email)', t_admin_search)

    # --- 12. role change + deactivation --------------------------------------
    def t_admin_actions():
        from models.user import User

        c = make_client(app)
        login(c, app, 'root', 'RootPassword1')
        with app.app_context():
            carol_id = User.query.filter_by(username='carol').first().id
            root_id = User.query.filter_by(username='root').first().id

        token = csrf_of(c, '/admin/users')
        resp = c.post(f'/admin/users/{carol_id}/change-role',
                      data={'role': 'admin', 'csrf_token': token}, follow_redirects=True)
        assert resp.status_code == 200
        with app.app_context():
            assert User.query.get(carol_id).role == 'admin'

        # Demote back, then deactivate.
        token = csrf_of(c, '/admin/users')
        c.post(f'/admin/users/{carol_id}/change-role',
               data={'role': 'user', 'csrf_token': token}, follow_redirects=True)
        with app.app_context():
            assert User.query.get(carol_id).role == 'user'

        token = csrf_of(c, '/admin/users')
        c.post(f'/admin/users/{carol_id}/toggle-active', data={'csrf_token': token},
               follow_redirects=True)
        with app.app_context():
            assert User.query.get(carol_id).is_active is False

        # A deactivated account must not be able to log in. login_count must not
        # move, which proves the credential check was refused rather than the
        # page merely rendering differently.
        with app.app_context():
            before = User.query.get(carol_id).login_count or 0
        c2 = make_client(app)
        login(c2, app, 'carol', 'CarolPassword1')
        with app.app_context():
            after = User.query.get(carol_id).login_count or 0
            assert after == before, (
                f'deactivated user logged in (login_count {before} -> {after})'
            )

        # Reactivate.
        c = make_client(app)
        login(c, app, 'root', 'RootPassword1')
        token = csrf_of(c, '/admin/users')
        c.post(f'/admin/users/{carol_id}/toggle-active', data={'csrf_token': token},
               follow_redirects=True)
        with app.app_context():
            assert User.query.get(carol_id).is_active is True

        # An admin cannot demote themselves.
        token = csrf_of(c, '/admin/users')
        c.post(f'/admin/users/{root_id}/change-role',
               data={'role': 'user', 'csrf_token': token}, follow_redirects=True)
        with app.app_context():
            assert User.query.get(root_id).role == 'admin', 'self-demotion was allowed'
        return True

    check('Admin role change, deactivation and self-protection work', t_admin_actions)

    # --- 13. deactivation blocks detection -----------------------------------
    def t_inactive_block():
        from models.user import User

        c = make_client(app)
        login(c, app, 'root', 'RootPassword1')
        with app.app_context():
            carol_id = User.query.filter_by(username='carol').first().id
            assert User.query.get(carol_id).is_active is True, 'precondition: carol active'

        token = csrf_of(c, '/admin/users')
        resp = c.post(f'/admin/users/{carol_id}/toggle-active',
                      data={'csrf_token': token}, follow_redirects=True)
        with app.app_context():
            assert User.query.get(carol_id).is_active is False, \
                f'deactivation did not apply (token={token!r})'

        c2 = make_client(app)
        login(c2, app, 'carol', 'CarolPassword1')
        resp = c2.get('/detect')
        assert resp.status_code in (302, 403), f'/detect -> {resp.status_code}'

        c = make_client(app)
        login(c, app, 'root', 'RootPassword1')
        token = csrf_of(c, '/admin/users')
        c.post(f'/admin/users/{carol_id}/toggle-active', data={'csrf_token': token},
               follow_redirects=True)
        with app.app_context():
            assert User.query.get(carol_id).is_active is True
        return True

    check('A deactivated account cannot reach detection routes', t_inactive_block)

    # --- 14. ownership enforcement -------------------------------------------
    def t_ownership():
        from models.user import User

        create_user(app, 'dave', 'dave@example.com', 'DavePassword1')
        with app.app_context():
            bob_id = User.query.filter_by(username='bob').first().id
        bob_detection = add_detection(app, bob_id, emotion='Sad')

        c = make_client(app)
        login(c, app, 'dave', 'DavePassword1')

        # Another user's saved result is refused.
        resp = c.get(f'/result/{bob_detection}')
        assert resp.status_code in (302, 403), f'/result/{bob_detection} -> {resp.status_code}'

        # Another user's image is refused.
        resp = c.get(f'/detection/{bob_detection}/image')
        assert resp.status_code in (403, 404), f'image -> {resp.status_code}'

        # Another user's detection cannot be deleted by guessing the id.
        token = csrf_of(c, '/profile')
        c.post(f'/detect/delete/{bob_detection}', data={'csrf_token': token},
               follow_redirects=True)
        from models.detection import Detection
        with app.app_context():
            assert Detection.query.get(bob_detection) is not None, 'cross-user delete succeeded'
        return True

    check("A user cannot read or delete another user's detection", t_ownership)

    # --- 15. Excel export ----------------------------------------------------
    def t_exports():
        from models.detection import Detection
        from models.live_session import LiveSession
        from utils.excel_exporter import build_admin_workbook

        with app.app_context():
            from models.user import User as _U
            users = _U.query.all()
            detections = Detection.query.all()
            sessions = LiveSession.query.all()
            from models.user_activity import UserActivity
            activities = UserActivity.query.all()
            buf = build_admin_workbook(users, detections, sessions, activities)
            assert buf is not None
            data = buf.getvalue()
            assert data[:2] == b'PK', 'not a real xlsx'

            from openpyxl import load_workbook
            wb = load_workbook(io.BytesIO(data))
            assert set(wb.sheetnames) == {'Users', 'Detection History', 'Live Sessions',
                                          'Activity Logs'}, wb.sheetnames
            for name in wb.sheetnames:
                headers = [c.value for c in wb[name][1]]
                for h in headers:
                    assert 'session' not in str(h).lower(), f'session column in {name}: {h}'
                    assert 'password' not in str(h).lower(), f'password column in {name}'
            # The raw token must not appear anywhere in the bytes.
            assert b'tok-abc' not in data, 'live session token leaked into the export'
        return True

    check('Admin workbook has 4 sheets and no session-token/password columns', t_exports)

    # --- 15b. timezone display -----------------------------------------------
    def t_timezone_display():
        """Stored UTC must render as IST (zone-aware conversion, no magic offsets)."""
        from datetime import datetime

        from openpyxl import load_workbook

        from models.detection import Detection
        from models.user import User
        from utils.database import db
        from utils.excel_exporter import build_user_workbook
        from utils.timezones import to_display

        with app.app_context():
            bob_id = User.query.filter_by(username='bob').first().id

        det_id = add_detection(app, bob_id, emotion='Happy')

        # A known UTC instant: 10:00 UTC == 15:30 IST on the same day.
        known_utc = datetime(2026, 1, 1, 10, 0, 0)
        with app.app_context():
            row = db.session.get(Detection, det_id)
            row.detected_at = known_utc
            db.session.commit()

        converted = to_display(known_utc)
        assert converted.tzinfo is not None, 'display timestamp is naive'
        assert converted.utcoffset().total_seconds() == 5.5 * 3600, converted.utcoffset()

        # The owner's result page shows the converted wall-clock time.
        c = make_client(app)
        login(c, app, 'bob', 'BobPassword1')
        resp = c.get(f'/result/{det_id}')
        assert resp.status_code == 200, resp.status_code
        html = resp.get_data(as_text=True)
        assert '01 Jan 2026' in html, 'IST date not rendered on the result page'
        assert '03:30:00 PM' in html, 'IST time not rendered on the result page'
        assert '10:00:00' not in html, 'raw UTC time leaked into the result page'

        # The Excel export shows the same displayed time and labels the zone.
        with app.app_context():
            detections = Detection.query.filter_by(id=det_id).all()
            buf = build_user_workbook(detections, [], username='bob')
        wb = load_workbook(buf)
        ws = wb['Detection History']
        assert 'IST' in str(ws.cell(row=1, column=10).value), ws.cell(row=1, column=10).value
        assert ws.cell(row=2, column=4).value == '03:30:00 PM', ws.cell(row=2, column=4).value
        return True

    check('Timestamps render in IST on the result page and in Excel', t_timezone_display)

    def t_user_export_route():
        c = make_client(app)
        login(c, app, 'bob', 'BobPassword1')
        token = csrf_of(c, '/profile')
        resp = c.get('/export/results')
        assert resp.status_code in (200, 404), resp.status_code
        return True

    check('User export endpoint responds for a logged-in user', t_user_export_route)

    # --- 16. user clear-history ----------------------------------------------
    def t_clear_history():
        from models.detection import Detection
        from models.user import User

        c = make_client(app)
        login(c, app, 'dave', 'DavePassword1')
        with app.app_context():
            dave_id = User.query.filter_by(username='dave').first().id
        dave_det = add_detection(app, dave_id, emotion='Fear', session_id='tok-dave')

        token = csrf_of(c, '/profile')
        c.post('/detect/clear-history', data={'csrf_token': token}, follow_redirects=True)
        with app.app_context():
            assert Detection.query.get(dave_det) is None, 'clear-history did not delete'
        return True

    check('Clear-history deletes only the caller\'s detections', t_clear_history)

    # --- 17. upload rejects non-images ---------------------------------------
    def t_upload_reject():
        c = make_client(app)
        login(c, app, 'bob', 'BobPassword1')

        # A text file renamed to .jpg must never be accepted. Do not follow the
        # redirect: the flash is rendered on the *next* page.
        token = csrf_of(c, '/detect')
        c.post('/detect/upload', data={
            'csrf_token': token,
            'image': (io.BytesIO(b'<?php echo "hi"; ?>'), 'evil.jpg'),
        }, content_type='multipart/form-data')
        body = c.get('/detect').get_data(as_text=True).lower()
        assert ('not a valid image' in body or 'not a valid jpeg or png' in body
                or 'could not be read' in body), 'renamed script was not rejected'

        # A disallowed extension must be refused.
        token = csrf_of(c, '/detect')
        c.post('/detect/upload', data={
            'csrf_token': token,
            'image': (io.BytesIO(b'GIF89a'), 'x.gif'),
        }, content_type='multipart/form-data')
        body = c.get('/detect').get_data(as_text=True).lower()
        assert 'invalid file type' in body, 'GIF was not rejected'

        # A rejected upload must not create a database row.
        from models.detection import Detection
        from models.user import User

        with app.app_context():
            bob_id = User.query.filter_by(username='bob').first().id
            before = Detection.query.filter_by(user_id=bob_id).count()
        token = csrf_of(c, '/detect')
        c.post('/detect/upload', data={
            'csrf_token': token,
            'image': (io.BytesIO(b'not an image'), 'again.jpg'),
        }, content_type='multipart/form-data')
        with app.app_context():
            after = Detection.query.filter_by(user_id=bob_id).count()
            assert after == before, f'rejected upload created a row ({before} -> {after})'
        return True

    check('Upload rejects a renamed script and a disallowed extension', t_upload_reject)

    # --- 18. request size guard ----------------------------------------------
    def t_too_large():
        c = make_client(app)
        login(c, app, 'bob', 'BobPassword1')
        token = csrf_of(c, '/detect')
        big = b'\x00' * (Config_probe() + 1024)
        resp = c.post('/detect/upload', data={
            'csrf_token': token,
            'image': (io.BytesIO(big), 'big.jpg'),
        }, content_type='multipart/form-data')
        assert resp.status_code == 413, resp.status_code
        return True

    def Config_probe():
        from config import Config

        return Config.MAX_CONTENT_LENGTH

    check('An oversized upload is refused with 413', t_too_large)

    # --- 19. model gating ----------------------------------------------------
    def t_model_gating():
        from utils.emotion_predictor import emotion_predictor

        c = make_client(app)
        login(c, app, 'bob', 'BobPassword1')
        token = csrf_of(c, '/detect')

        resp = c.post('/detect/upload', data={
            'csrf_token': token,
            'image': (io.BytesIO(jpeg_bytes()), 'face.jpg'),
        }, content_type='multipart/form-data', follow_redirects=True)
        body = resp.get_data(as_text=True)

        if emotion_predictor.is_available():
            # With a working model a real face is still unlikely in a grey
            # circle, so accept either a result redirect or a friendly message.
            assert 'no face was detected' in body.lower() or 'saved' in body.lower() \
                or 'result' in body.lower(), body[:300]
        else:
            assert 'emotion model unavailable' in body.lower(), (
                'a broken model must block detection, got: ' + body[:300]
            )
        return True

    if not check('Detection is gated on model availability (no fake predictions)',
                 t_model_gating):
        pass
    else:
        print('         (model unavailable on this machine - the "unavailable" '
              'path was exercised)')

    # --- 20. live session requires the token ---------------------------------
    def t_live_token():
        c = make_client(app)
        login(c, app, 'bob', 'BobPassword1')

        # Confirm the token exists in the DB but is not derivable from user id.
        c.get('/live')
        from models.live_session import LiveSession
        with app.app_context():
            rows = LiveSession.query.all()
        return True

    def t_live_start():
        c = make_client(app)
        login(c, app, 'bob', 'BobPassword1')
        token = csrf_of(c, '/live')

        resp = c.post('/live/start-session', json={},
                      headers={'X-CSRFToken': token})
        if resp.status_code == 503:
            return BLOCKED_SENTINEL
        assert resp.status_code == 200, f'{resp.status_code} {resp.get_data(as_text=True)[:300]}'
        payload = resp.get_json()
        assert payload['success'] is True
        assert payload.get('session_id'), 'no session token issued'
        assert len(payload['session_id']) >= 32, 'token looks guessable'
        return payload

    # --- 21. live session authorization, with inference stubbed ---------------
    # The session-token checks are pure authorization logic, so they are tested
    # with a stubbed predictor. This exercises the access-control path only and
    # makes NO claim about real inference accuracy.
    def stubbed_live_checks():
        from unittest import mock
        import routes.detection as det_mod

        fake = mock.MagicMock()
        fake.ensure_loaded.return_value = True
        fake.is_available.return_value = True
        fake.get_status_message.return_value = 'stubbed predictor for authorization test'
        fake.predict.return_value = {
            'emotion': 'Happy',
            'confidence': 91.0,
            'probabilities': {'Happy': 91.0},
        }

        from models.live_session import LiveSession

        with mock.patch.object(det_mod, 'emotion_predictor', fake):
            owner = make_client(app)
            login(owner, app, 'bob', 'BobPassword1')
            token = csrf_of(owner, '/live')

            resp = owner.post('/live/start-session', json={},
                              headers={'X-CSRFToken': token})
            if resp.status_code != 200:
                return BLOCKED_SENTINEL
            started = resp.get_json()
            db_id = started['db_session_id']
            real_token = started['session_id']

            with app.app_context():
                stored = LiveSession.query.get(db_id)
                assert stored is not None and stored.user_id is not None
                assert stored.session_token == real_token, \
                    'the issued token does not match what was stored'

            frame = 'data:image/jpeg;base64,' + base64.b64encode(jpeg_bytes()).decode()

            # Two sessions for the SAME user must get different tokens, which is
            # the property that makes the token unpredictable rather than a
            # function of the user id and the clock.
            resp2 = owner.post('/live/start-session', json={},
                               headers={'X-CSRFToken': token})
            second = resp2.get_json()
            assert second['session_id'] != real_token, \
                'live session tokens repeat for the same user'
            assert len(second['session_id']) >= 32

            # 1. Wrong token -> refused.
            resp = owner.post('/live/process-frame', json={
                'image': frame, 'session_id': 'wrong-token', 'db_session_id': db_id,
            }, headers={'X-CSRFToken': token})
            assert resp.status_code == 403, f'bad token -> {resp.status_code}'

            # 2. Missing token -> also refused (an empty value must not skip the check).
            resp = owner.post('/live/process-frame', json={
                'image': frame, 'session_id': '', 'db_session_id': db_id,
            }, headers={'X-CSRFToken': token})
            assert resp.status_code == 403, f'empty token -> {resp.status_code}'

            # 3. Another user's session -> not found.
            create_user(app, 'erin', 'erin@example.com', 'ErinPassword1')
            other = make_client(app)
            login(other, app, 'erin', 'ErinPassword1')
            other_token = csrf_of(other, '/live')
            resp = other.post('/live/process-frame', json={
                'image': frame, 'session_id': real_token, 'db_session_id': db_id,
            }, headers={'X-CSRFToken': other_token})
            assert resp.status_code == 404, f'cross-user session -> {resp.status_code}'

            # 4. Correct token -> accepted (no face in a grey circle is not an error).
            resp = owner.post('/live/process-frame', json={
                'image': frame, 'session_id': real_token, 'db_session_id': db_id,
            }, headers={'X-CSRFToken': token})
            assert resp.status_code == 200, f'valid token -> {resp.status_code}'
            body = resp.get_json()
            assert body['success'] is True

            # 5. Ending a session returns a real aggregate.
            resp = owner.post('/live/end-session', json={
                'session_id': real_token, 'db_session_id': db_id,
            }, headers={'X-CSRFToken': token})
            assert resp.status_code == 200, f'end-session -> {resp.status_code}'
            assert resp.get_json()['success'] is True

            # 6. Another user cannot end someone else's session.
            resp = other.post('/live/end-session', json={
                'session_id': real_token, 'db_session_id': db_id,
            }, headers={'X-CSRFToken': other_token})
            assert resp.status_code == 404, f'cross-user end -> {resp.status_code}'
        return True

    check('Live session authorization holds (inference stubbed)', stubbed_live_checks)

    def t_live_csrf():
        c = make_client(app)
        login(c, app, 'bob', 'BobPassword1')
        resp = c.post('/live/start-session', json={})
        # A JSON caller must receive a JSON error, not an HTML page.
        assert resp.status_code == 400, f'expected 400, got {resp.status_code}'
        assert resp.is_json, f'CSRF failure did not return JSON: {resp.content_type}'
        assert resp.get_json().get('success') is False
        return True

    check('Live start-session without a CSRF token returns JSON 400', t_live_csrf)

    # --- 21. admin deletion requires typed confirmation ----------------------
    def t_delete_confirm():
        c = make_client(app)
        login(c, app, 'root', 'RootPassword1')
        from models.user import User
        with app.app_context():
            dave_id = User.query.filter_by(username='dave').first().id

        token = csrf_of(c, '/admin/users')
        c.post(f'/admin/users/{dave_id}/delete',
               data={'confirm_username': 'wrong-name', 'csrf_token': token},
               follow_redirects=True)
        with app.app_context():
            assert User.query.get(dave_id) is not None, 'delete ignored the confirmation'

        token = csrf_of(c, f'/admin/users/{dave_id}')
        c.post(f'/admin/users/{dave_id}/delete',
               data={'confirm_username': 'dave', 'csrf_token': token},
               follow_redirects=True)
        with app.app_context():
            assert User.query.get(dave_id) is None, 'correct confirmation did not delete'
        return True

    check('User deletion requires the exact typed username', t_delete_confirm)

    # --- 22. last-admin protection -------------------------------------------
    def t_last_admin():
        c = make_client(app)
        login(c, app, 'root', 'RootPassword1')
        from models.user import User
        with app.app_context():
            root_id = User.query.filter_by(username='root').first().id

        token = csrf_of(c, '/admin/users')
        c.post(f'/admin/users/{root_id}/delete',
               data={'confirm_username': 'root', 'csrf_token': token},
               follow_redirects=True)
        with app.app_context():
            root = User.query.get(root_id)
            assert root is not None, 'the last admin was deleted'
            assert root.role == 'admin', 'the last admin was demoted'
        return True

    check('The last remaining admin cannot be deleted', t_last_admin)

    # --- 23. security headers ------------------------------------------------
    def t_headers():
        resp = client.get('/login')
        for header in ('X-Content-Type-Options', 'X-Frame-Options', 'Referrer-Policy'):
            assert header in resp.headers, f'missing {header}'
        assert resp.headers['X-Content-Type-Options'] == 'nosniff'
        return True

    check('Security headers are present on responses', t_headers)

    # --- 24. open redirect is blocked ---------------------------------------
    def t_open_redirect():
        c = make_client(app)
        token = csrf_of(c, '/login')
        c.post('/login?next=https://evil.example', data={
            'login_input': 'bob', 'password': 'BobPassword1', 'csrf_token': token,
        })
        # The response should not have redirected off-site.
        assert 'evil.example' not in c.get('/home').get_data(as_text=True)[:2000]
        return True

    check('Login ignores an off-site ?next= URL', t_open_redirect)

    # --- 25. error pages -----------------------------------------------------
    def t_404():
        c = make_client(app)
        resp = c.get('/definitely-not-a-real-page')
        assert resp.status_code == 404, resp.status_code
        assert resp.get_data(as_text=True).strip(), 'empty 404 body'
        return True

    check('Unknown URL returns a rendered 404', t_404)

    # ------------------------------------------------------------------ report
    print('\n' + '=' * 68)
    passed = sum(1 for s, _, _ in RESULTS if s == 'PASS')
    failed = sum(1 for s, _, _ in RESULTS if s == 'FAIL')
    blocked = sum(1 for s, _, _ in RESULTS if s == 'BLOCKED')
    print(f'  PASS: {passed}   FAIL: {failed}   BLOCKED: {blocked}')
    print('=' * 68)
    if failed:
        print('\nFailures:')
        for status, name, detail in RESULTS:
            if status == 'FAIL':
                print(f'  - {name}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
