# =============================================================================
# Database Utility
# =============================================================================
# Creates the shared SQLAlchemy instance and wires it to the Flask app.
#
# The database backend is chosen purely from configuration:
#
#   DATABASE_URL set      -> PostgreSQL (psycopg 3 driver)
#   DATABASE_URL empty    -> SQLite file under ./database/
#
# `db` lives here (not in app.py) so that models can import it without
# creating a circular import.
# =============================================================================

import logging
import os

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

logger = logging.getLogger(__name__)

db = SQLAlchemy()


def database_kind():
    """Return 'postgresql' or 'sqlite' for the configured database."""
    from config import Config

    return Config.DB_DIALECT


def is_postgres():
    """True when the app is configured to talk to PostgreSQL."""
    return database_kind() == 'postgresql'


def init_db(app):
    """
    Bind SQLAlchemy to the Flask application.

    On SQLite the local database directory is created first. Missing tables are
    created afterwards, but ``create_all()`` never drops or alters an existing
    table, so this is safe to run on every boot. Schema *changes* on an existing
    PostgreSQL database should go through Alembic/Flask-Migrate instead.
    """
    from config import Config

    if not is_postgres():
        os.makedirs(Config.DATABASE_DIR, exist_ok=True)

    db.init_app(app)

    # Import every model so SQLAlchemy has the complete metadata.
    import models  # noqa: F401

    with app.app_context():
        if Config.AUTO_CREATE_TABLES:
            try:
                db.create_all()
            except Exception as exc:  # pragma: no cover - depends on DB state
                # Do not crash the import: surface the problem and continue so
                # that /health and the admin UI can report it.
                db.session.rollback()
                logger.error('create_all() failed: %s', exc)
        else:
            logger.info('AUTO_CREATE_TABLES disabled; run migrations instead')

    log_database_summary(app)


def log_database_summary(app=None):
    """
    Log the selected database type.

    Only the *type* is logged. The connection string, and therefore any
    password inside it, is never printed.
    """
    from config import Config, redact_database_url

    if is_postgres():
        logger.info('Database: POSTGRESQL')
    else:
        logger.info('Database: SQLITE (%s)', os.path.join('database', Config.DATABASE_FILENAME))
    logger.info('Storage: %s', _storage_backend_label())


def _storage_backend_label():
    """Credential-free storage backend label."""
    try:
        from utils.storage import describe_storage

        return describe_storage()
    except Exception:
        return 'UNKNOWN'


def database_health():
    """
    Lightweight connectivity probe.

    Returns a dict that is safe to expose: it reports *whether* the database
    answered, never how it is addressed.
    """
    try:
        db.session.execute(text('SELECT 1'))
        db.session.rollback()
        return {'ok': True, 'engine': database_kind()}
    except Exception as exc:
        db.session.rollback()
        logger.warning('Database health check failed: %s', exc)
        return {'ok': False, 'engine': database_kind()}
