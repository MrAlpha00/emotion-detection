# =============================================================================
# Database Utility
# =============================================================================
# Creates the SQLAlchemy instance and provides database initialization.
# The db instance is created here (not in app.py) to avoid circular imports
# — models import db from here, and app.py initializes db with the app.
# =============================================================================

from flask_sqlalchemy import SQLAlchemy

# Global SQLAlchemy instance — initialized with the Flask app in app.py
db = SQLAlchemy()


def init_db(app):
    """
    Initialize the database with the Flask application.
    Creates all tables if they don't exist yet.
    Also ensures the database directory exists.
    """
    import os
    from config import Config

    # Ensure the database directory exists
    os.makedirs(Config.DATABASE_DIR, exist_ok=True)

    # Bind SQLAlchemy to the Flask app
    db.init_app(app)

    # Create all tables within the application context
    with app.app_context():
        # Import all models so SQLAlchemy knows about them
        from models import User, Detection, LiveSession  # noqa: F401
        db.create_all()
        print(f"[Database] Initialized at: {Config.DATABASE_PATH}")
