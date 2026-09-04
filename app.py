# =============================================================================
# Emotion Detection Using Facial Expression — Main Application
# =============================================================================
# Entry point for the Flask application.
# Initializes all components: database, blueprints, login manager,
# emotion predictor, and template context processors.
# =============================================================================

import os
from flask import Flask, render_template
from flask_login import LoginManager
from config import Config
from utils.database import db, init_db


def create_app():
    """Application factory — creates and configures the Flask app."""

    app = Flask(__name__)
    app.config.from_object(Config)

    # -------------------------------------------------------------------------
    # Ensure required directories exist
    # -------------------------------------------------------------------------
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.RESULTS_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(Config.BASE_DIR, 'model'), exist_ok=True)

    # -------------------------------------------------------------------------
    # Initialize Database
    # -------------------------------------------------------------------------
    init_db(app)

    # -------------------------------------------------------------------------
    # Initialize Flask-Login
    # -------------------------------------------------------------------------
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'warning'

    @login_manager.user_loader
    def load_user(user_id):
        from models.user import User
        return User.query.get(int(user_id))

    # -------------------------------------------------------------------------
    # Initialize Emotion Predictor (loads model once at startup)
    # -------------------------------------------------------------------------
    from utils.emotion_predictor import emotion_predictor
    with app.app_context():
        emotion_predictor.load_model()

    # -------------------------------------------------------------------------
    # Template Context Processors
    # -------------------------------------------------------------------------
    @app.context_processor
    def inject_globals():
        """Make common variables available in all templates."""
        from datetime import datetime, timezone
        return {
            'now': datetime.now(timezone.utc),
            'model_available': emotion_predictor.is_available(),
            'model_status_message': emotion_predictor.get_status_message(),
            'app_config': Config,
        }

    # -------------------------------------------------------------------------
    # Register Blueprints
    # -------------------------------------------------------------------------
    from routes.auth import auth_bp
    from routes.dashboard import dashboard_bp
    from routes.detection import detection_bp
    from routes.export import export_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(detection_bp)
    app.register_blueprint(export_bp)

    # -------------------------------------------------------------------------
    # Error Handlers
    # -------------------------------------------------------------------------
    @app.errorhandler(404)
    def not_found(e):
        return render_template('error.html',
                               error_code=404,
                               error_title='Page Not Found',
                               error_message='The page you are looking for does not exist.'), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template('error.html',
                               error_code=500,
                               error_title='Server Error',
                               error_message='An internal server error occurred. Please try again later.'), 500

    @app.errorhandler(413)
    def file_too_large(e):
        return render_template('error.html',
                               error_code=413,
                               error_title='File Too Large',
                               error_message='The uploaded file exceeds the maximum allowed size of 16 MB.'), 413

    return app


# =============================================================================
# Run the application
# =============================================================================
if __name__ == '__main__':
    app = create_app()
    print("\n" + "=" * 60)
    print("  Emotion Detection Using Facial Expression")
    print("  Running at: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=Config.DEBUG, host='127.0.0.1', port=5000)
