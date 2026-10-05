# =============================================================================
# Emotion Detection Using Facial Expression - Application Entry Point
# =============================================================================
# Application factory, extension wiring, error handlers and the health endpoint.
#
# Deployment
# ----------
#   Local      ->  python app.py       (SQLite + local filesystem storage)
#   Vercel     ->  imports `app`       (PostgreSQL + Supabase Storage)
#
# The module-level `app` object at the bottom of this file is the WSGI callable
# Vercel looks for. `app.run()` is only ever reached under
# `if __name__ == '__main__':`, so importing this file on a serverless host
# starts the development server.
# =============================================================================

import logging
import os
import sys

from flask import Flask, jsonify, render_template, request
from flask_login import LoginManager
from werkzeug.exceptions import HTTPException

from config import Config, get_config

# Make the project root importable when this file is executed directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

# Reduce noise from third-party libraries.
logging.getLogger('werkzeug').setLevel(logging.WARNING)


def create_app(config_object=None):
    """
    Build and configure the Flask application.

    Args:
        config_object: optional config class overriding environment detection.
    """
    app = Flask(__name__)

    config_class = config_object or get_config()

    # Instantiate rather than only reading the class, so ProductionConfig's
    # fail-fast validation (SECRET_KEY, DATABASE_URL, Supabase key) actually
    # runs before the database or the filesystem is touched.
    config_instance = config_class()

    app.config.from_object(config_instance)
    Config = config_class  # noqa: N806 - shadow to keep routes importing Config

    # Make the configured config importable as `config.Config` for the modules
    # that reference it directly (utils, routes, templates).
    import config as config_module

    config_module.Config = config_class

    # -------------------------------------------------------------------------
    # Directory setup
    # -------------------------------------------------------------------------
    # Only the local (SQLite) configuration touches the filesystem. A production
    # PostgreSQL + Supabase deployment must not create local paths at all, which
    # matters on serverless hosts where the bundle is read-only.
    if config_class.USE_SQLITE:
        try:
            os.makedirs(config_class.UPLOAD_FOLDER, exist_ok=True)
            os.makedirs(config_class.RESULTS_FOLDER, exist_ok=True)
            os.makedirs(config_class.DATABASE_DIR, exist_ok=True)
        except OSError as exc:
            # Read-only filesystem (possible on some hosts) is not fatal.
            logger.warning('Could not create local directories: %s', exc)

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------
    from utils.database import db, init_db, log_database_summary

    init_db(app)

    # -------------------------------------------------------------------------
    # Flask-WTF (CSRF)
    # -------------------------------------------------------------------------
    # Every state-changing form and every JSON POST carries a token. The
    # extension is initialised here so `csrf_token()` is available in all
    # templates, including the error pages rendered outside a blueprint.
    from flask_wtf.csrf import CSRFError, CSRFProtect

    csrf = CSRFProtect()
    csrf.init_app(app)

    # The live-detection and admin JSON endpoints are authenticated with the
    # session cookie, so they need the token too. The client reads it from the
    # `<meta name="csrf-token">` tag rendered in base.html.
    @app.context_processor
    def inject_csrf_token():
        from flask_wtf.csrf import generate_csrf

        return {'csrf_token_value': generate_csrf()}

    @app.errorhandler(CSRFError)
    def handle_csrf_error(error):
        # A rejected token is a security event, not a crash. Log it without the
        # submitted value and never reveal what was expected.
        logger.warning(
            'CSRF validation failed for %s %s (reason: %s)',
            request.method if request else '?',
            request.path if request else '?',
            getattr(error, 'reason', 'unknown'),
        )

        # JSON callers (the live-detection endpoints and any fetch() call) must
        # get JSON back, not an HTML error page they cannot parse.
        if _wants_json():
            return jsonify({
                'success': False,
                'error': 'CSRF token missing or invalid. Reload the page and try again.',
            }), 400

        return render_template('403.html', csrf_error=True), 403

    # -------------------------------------------------------------------------
    # Flask-Login
    # -------------------------------------------------------------------------
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'warning'
    # Deactivated accounts are caught by @active_user_required; this keeps the
    # framework from re-deriving the view for anonymous users too.
    login_manager.session_protection = 'strong'

    @login_manager.user_loader
    def load_user(user_id):
        from models.user import User

        try:
            return db.session.get(User, int(user_id))
        except (TypeError, ValueError):
            return None

    # -------------------------------------------------------------------------
    # Flask-Migrate (Alembic)
    # -------------------------------------------------------------------------
    # Registered but never auto-upgraded. Schema changes are an explicit
    # `flask db upgrade` step, so deploying can never silently mutate a
    # production schema.
    try:
        from flask_migrate import Migrate

        migrations_dir = os.path.join(config_class.BASE_DIR, 'migrations')
        Migrate(app, db, directory=migrations_dir)
    except Exception as exc:
        logger.warning('Flask-Migrate not initialised: %s', exc)

    # -------------------------------------------------------------------------
    # Blueprints
    # -------------------------------------------------------------------------
    from routes.admin import admin_bp
    from routes.auth import auth_bp
    from routes.dashboard import dashboard_bp
    from routes.detection import detection_bp
    from routes.export import export_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(detection_bp)
    app.register_blueprint(export_bp)
    app.register_blueprint(admin_bp)

    # -------------------------------------------------------------------------
    # Template context
    # -------------------------------------------------------------------------
    @app.context_processor
    def inject_globals():
        from utils.emotion_predictor import emotion_predictor

        return {
            'now': datetime_utcnow(),
            'model_available': emotion_predictor.is_available(),
            'model_status_message': emotion_predictor.get_status_message(),
            'emotion_labels': Config.EMOTION_LABELS,
            'app_config': Config,
            'storage_backend': Config.STORAGE_BACKEND,
            'is_admin': bool(current_user_is_admin()),
        }

    # -------------------------------------------------------------------------
    # Security headers
    # -------------------------------------------------------------------------
    @app.after_request
    def set_security_headers(response):
        response.headers.setdefault('X-Content-Type-Options', 'nosniff')
        response.headers.setdefault('X-Frame-Options', 'SAMEORIGIN')
        response.headers.setdefault('Referrer-Policy', 'strict-origin-when-cross-origin')
        response.headers.setdefault('Permissions-Policy', 'camera=(self), microphone=()')
        if not config_class.DEBUG:
            response.headers.setdefault(
                'Strict-Transport-Security', 'max-age=31536000; includeSubDomains'
            )
        return response

    # -------------------------------------------------------------------------
    # Request size guard
    # -------------------------------------------------------------------------
    @app.before_request
    def enforce_content_length():
        from flask import request

        limit = config_class.MAX_CONTENT_LENGTH
        declared = request.content_length
        if declared is not None and declared > limit:
            from flask import abort

            abort(413)
        return None

    # -------------------------------------------------------------------------
    # Error handlers
    # -------------------------------------------------------------------------
    def _wants_json():
        """
        Decide whether the caller expects a JSON body rather than an HTML page.

        Checked in order of reliability:
          1. /health always gets JSON
          2. an explicit ``Accept: application/json``
          3. a JSON request body (``Content-Type: application/json``), which is
             what every fetch() call in this project sends
          4. the XHR marker
        ``Accept: */*`` alone is NOT treated as a JSON request, because browsers
        send that by default even when navigating to an HTML page.
        """
        from flask import request

        if request.path == '/health':
            return True

        accept = request.accept_mimetypes
        if accept.accept_json and not (accept.accept_html or accept.accept_xhtml):
            return True

        if request.mimetype == 'application/json':
            return True

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return True

        return False

    @app.errorhandler(400)
    def bad_request(error):
        if _wants_json():
            return jsonify({'error': 'Bad request'}), 400
        return render_template('error.html', error_code=400,
                               error_title='Bad Request',
                               error_message='The request could not be understood.'), 400

    @app.errorhandler(401)
    def unauthorized(error):
        if _wants_json():
            return jsonify({'error': 'Authentication required'}), 401
        return render_template('error.html', error_code=401,
                               error_title='Authentication Required',
                               error_message='Please log in to continue.'), 401

    @app.errorhandler(403)
    def forbidden(error):
        if _wants_json():
            return jsonify({'error': 'Forbidden'}), 403
        return render_template('403.html'), 403

    @app.errorhandler(404)
    def not_found(error):
        if _wants_json():
            return jsonify({'error': 'Not found'}), 404
        return render_template('404.html'), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        if _wants_json():
            return jsonify({'error': 'Method not allowed'}), 405
        return render_template('error.html', error_code=405,
                               error_title='Method Not Allowed',
                               error_message='That action is not available here.'), 405

    @app.errorhandler(413)
    def too_large(error):
        limit_mb = round(config_class.MAX_CONTENT_LENGTH / (1024 * 1024), 1)
        if _wants_json():
            return jsonify({'error': f'File too large. Maximum size is {limit_mb} MB.'}), 413
        return render_template('error.html', error_code=413,
                               error_title='File Too Large',
                               error_message=f'The upload exceeds the maximum allowed size of {limit_mb} MB.'), 413

    @app.errorhandler(429)
    def too_many_requests(error):
        retry_after = getattr(error, 'retry_after', None) or 60
        if _wants_json():
            response = jsonify({'error': 'Too many requests. Please slow down.'})
            response.status_code = 429
            response.headers['Retry-After'] = str(retry_after)
            return response
        response = render_template('error.html', error_code=429,
                                   error_title='Too Many Requests',
                                   error_message='Too many requests were sent in a short time. Please wait a moment and try again.')
        return response, 429

    @app.errorhandler(500)
    def internal_error(error):
        # Never expose a traceback or a database URL to the browser.
        logger.exception('Unhandled server error: %s', error)
        try:
            from utils.database import db

            db.session.rollback()
        except Exception:
            pass
        if _wants_json():
            return jsonify({'error': 'Internal server error'}), 500
        return render_template('error.html', error_code=500,
                               error_title='Server Error',
                               error_message='An internal server error occurred. Please try again later.'), 500

    @app.errorhandler(Exception)
    def handle_unexpected(error):
        # HTTP exceptions keep their own status; anything else is a 500.
        if isinstance(error, HTTPException):
            return error
        logger.exception('Unhandled exception: %s', error)
        try:
            from utils.database import db

            db.session.rollback()
        except Exception:
            pass
        if _wants_json():
            return jsonify({'error': 'Internal server error'}), 500
        return render_template('error.html', error_code=500,
                               error_title='Server Error',
                               error_message='An internal server error occurred. Please try again later.'), 500

    # -------------------------------------------------------------------------
    # Health check
    # -------------------------------------------------------------------------
    @app.route('/health')
    def health():
        """
        Liveness/readiness probe.

        Returns status, the configured database engine and whether the database
        answered. Never returns secrets, connection strings or API keys.
        """
        from utils.database import database_health, database_kind

        db_health = database_health()

        payload = {
            'status': 'ok' if db_health['ok'] else 'degraded',
            'database': {
                'engine': database_kind(),
                'connected': db_health['ok'],
            },
            'storage': Config.STORAGE_BACKEND,
        }

        # A degraded database still returns 200 so the platform does not kill a
        # cold instance over a transient blip; the body reports the truth.
        return jsonify(payload), 200

    # -------------------------------------------------------------------------
    # Startup logging (never prints secrets)
    # -------------------------------------------------------------------------
    with app.app_context():
        log_database_summary(app)

        from utils.emotion_predictor import emotion_predictor

        loaded = emotion_predictor.warm_up()
        logger.info('Emotion model: %s', 'AVAILABLE' if loaded else 'UNAVAILABLE')
        if not loaded:
            # Not fatal: detection routes will report "Emotion model unavailable".
            logger.warning('Emotion model unavailable: %s', emotion_predictor.get_status_message())

    logger.info(
        'Emotion Detection application ready (db=%s, storage=%s, debug=%s)',
        Config.DB_DIALECT.upper(),
        Config.STORAGE_BACKEND.upper(),
        Config.DEBUG,
    )

    return app


def datetime_utcnow():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


def current_user_is_admin():
    from flask_login import current_user

    try:
        return current_user.is_authenticated and current_user.role == 'admin'
    except Exception:
        return False


# =============================================================================
# WSGI entry point
# =============================================================================
# Vercel imports this module and uses `app` as the WSGI callable. Nothing below
# runs at import time other than creating the application object.
app = create_app()


if __name__ == '__main__':
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', '5000'))
    print('\n' + '=' * 64)
    print('  Emotion Detection Using Facial Expression')
    print(f'  Database : {Config.DB_DIALECT.upper()}')
    print(f'  Storage  : {Config.STORAGE_BACKEND.upper()}')
    print(f'  Running at: http://{host}:{port}')
    print('=' * 64 + '\n')
    # Local development only. Never executed on Vercel.
    app.run(host=host, port=port, debug=Config.DEBUG)
