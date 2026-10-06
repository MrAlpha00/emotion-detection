-- =============================================================================
-- supabase_schema.sql
--
-- Complete production schema for the emotion-detection application.
--
-- SOURCE OF TRUTH
-- ---------------
-- Alembic revision ca4a67aa2044:
--     migrations/versions/ca4a67aa2044_baseline_and_legacy_upgrade.py
-- Cross-checked against the SQLAlchemy models in models/.
--
-- Every CREATE TABLE / CREATE INDEX below is the exact DDL that revision
-- emits through SQLAlchemy's PostgreSQL dialect for a brand new database
-- (its _create_everything() path: _META.create_all() followed by the
-- _INDEXES loop).
--
-- HOW TO RUN
-- ----------
-- Run once against a fresh, empty Supabase project (SQL editor or psql).
-- The script is NOT idempotent: it creates objects unconditionally, so it
-- expects an empty `public` schema. Nothing outside these tables is touched.
--
-- NOTES
-- -----
-- * Timestamps are TIMESTAMP WITHOUT TIME ZONE, matching sa.DateTime() and
--   the documented contract in models/live_session.py (the application
--   always writes explicit UTC values).
-- * `users.role`, `users.is_active` and `users.login_count` carry server
--   defaults. They come from this revision's _NEW_COLUMNS entries, which
--   mirror the Python-side model defaults so the fresh-database path and
--   the legacy-SQLite-upgrade path converge on the same schema.
-- * The three foreign keys use ON DELETE CASCADE, exactly as the fresh
--   path creates them.
-- * alembic_version is stamped with this revision so a subsequent
--   `flask db upgrade` is a no-op instead of replaying the baseline.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- users
-- -----------------------------------------------------------------------------
CREATE TABLE users (
	id SERIAL NOT NULL,
	username VARCHAR(80) NOT NULL,
	email VARCHAR(120) NOT NULL,
	password_hash VARCHAR(256) NOT NULL,
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
	role VARCHAR(20) NOT NULL DEFAULT 'user',
	is_active BOOLEAN NOT NULL DEFAULT TRUE,
	login_count INTEGER NOT NULL DEFAULT 0,
	last_login_at TIMESTAMP WITHOUT TIME ZONE,
	PRIMARY KEY (id)
);

CREATE UNIQUE INDEX ix_users_username ON users (username);
CREATE UNIQUE INDEX ix_users_email ON users (email);
CREATE INDEX ix_users_role ON users (role);
CREATE INDEX ix_users_is_active ON users (is_active);


-- -----------------------------------------------------------------------------
-- detections
-- -----------------------------------------------------------------------------
CREATE TABLE detections (
	id SERIAL NOT NULL,
	user_id INTEGER NOT NULL,
	detection_type VARCHAR(20) NOT NULL,
	emotion VARCHAR(50) NOT NULL,
	confidence FLOAT NOT NULL,
	image_path VARCHAR(500),
	processed_image_path VARCHAR(500),
	detected_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
	session_id VARCHAR(100),
	face_count INTEGER,
	processing_time FLOAT,
	PRIMARY KEY (id),
	CONSTRAINT fk_detections_user_id FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE INDEX ix_detections_user_id ON detections (user_id);
CREATE INDEX ix_detections_detection_type ON detections (detection_type);
CREATE INDEX ix_detections_emotion ON detections (emotion);
CREATE INDEX ix_detections_detected_at ON detections (detected_at);
CREATE INDEX ix_detections_session_id ON detections (session_id);
CREATE INDEX ix_detections_user_detected ON detections (user_id, detected_at);
CREATE INDEX ix_detections_type_emotion ON detections (detection_type, emotion);


-- -----------------------------------------------------------------------------
-- live_sessions
-- -----------------------------------------------------------------------------
CREATE TABLE live_sessions (
	id SERIAL NOT NULL,
	user_id INTEGER NOT NULL,
	started_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
	ended_at TIMESTAMP WITHOUT TIME ZONE,
	duration_seconds INTEGER,
	dominant_emotion VARCHAR(50),
	average_confidence FLOAT,
	total_detections INTEGER,
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
	session_token VARCHAR(100),
	PRIMARY KEY (id),
	CONSTRAINT fk_live_sessions_user_id FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE INDEX ix_live_sessions_user_id ON live_sessions (user_id);
CREATE INDEX ix_live_sessions_started_at ON live_sessions (started_at);
CREATE INDEX ix_live_sessions_session_token ON live_sessions (session_token);
CREATE INDEX ix_live_sessions_user_started ON live_sessions (user_id, started_at);


-- -----------------------------------------------------------------------------
-- user_activities
-- -----------------------------------------------------------------------------
CREATE TABLE user_activities (
	id SERIAL NOT NULL,
	user_id INTEGER,
	activity_type VARCHAR(50) NOT NULL,
	details VARCHAR(500),
	ip_address VARCHAR(45),
	user_agent VARCHAR(500),
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
	PRIMARY KEY (id),
	CONSTRAINT fk_user_activities_user_id FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE INDEX ix_user_activities_user_id ON user_activities (user_id);
CREATE INDEX ix_user_activities_activity_type ON user_activities (activity_type);
CREATE INDEX ix_user_activities_created_at ON user_activities (created_at);
CREATE INDEX ix_user_activities_user_created ON user_activities (user_id, created_at);


-- -----------------------------------------------------------------------------
-- alembic_version (Alembic's own revision-tracking table)
--
-- Shape is Alembic's standard one for a single-version history. The row
-- stamps revision ca4a67aa2044 so the baseline is recorded as applied.
-- -----------------------------------------------------------------------------
CREATE TABLE alembic_version (
	version_num VARCHAR(32) NOT NULL,
	CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);

INSERT INTO alembic_version (version_num) VALUES ('ca4a67aa2044');
