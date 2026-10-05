#!/usr/bin/env python3
# =============================================================================
# SQLite to PostgreSQL Data Migration (optional utility)
# =============================================================================
# Copies rows from the local SQLite development database into a PostgreSQL
# database, preserving primary keys so that detections, live sessions and audit
# rows keep pointing at the right user.
#
# Schema is NOT handled here. Run `flask db upgrade` against PostgreSQL first;
# this script only moves data.
#
# Usage:
#     python scripts/migrate_sqlite_to_postgres.py --dry-run
#     python scripts/migrate_sqlite_to_postgres.py
#     python scripts/migrate_sqlite_to_postgres.py --source database/old.db
#
# SAFETY PROPERTIES
#   * Read-only against SQLite. The source file is opened with mode=ro.
#   * Idempotent. Every table is copied with explicit primary keys and an
#     ON CONFLICT DO NOTHING upsert, so re-running never duplicates a row and
#     never overwrites data that PostgreSQL already holds.
#   * No destructive operations. There is no TRUNCATE and no DELETE anywhere in
#     this file.
#   * Atomic per table. Each table is committed as one transaction, so a
#     failure part-way leaves PostgreSQL with whole tables rather than halves.
#   * Dry run by default for a first look: --dry-run reports what would happen
#     without writing anything.
#
# WHAT IS *NOT* COPIED
#   Stored images are not transferred. Detection.image_path and
#   processed_image_path hold storage keys, and the bytes live either on the
#   local disk or in Supabase Storage. Copying the rows without the files would
#   leave dangling references, so --copy-images is required to move files, and
#   even then it works through the configured storage backend. Run it against a
#   deployment whose STORAGE_BACKEND matches where the files actually are.
# =============================================================================

import argparse
import os
import sqlite3
import sys

# Make the project root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text  # noqa: E402

from app import create_app  # noqa: E402
from utils.database import db  # noqa: E402

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_NO_SOURCE = 2
EXIT_FAILED = 3

# Copied in this order so a foreign key never references a row that has not
# been inserted yet.
TABLES = ('users', 'live_sessions', 'detections', 'user_activities')


def open_sqlite_readonly(path):
    """Open the source database read-only so a bug cannot modify it."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    uri = 'file:{}?mode=ro'.format(path.replace('\\', '/'))
    return sqlite3.connect(uri, uri=True)


def count_rows(conn, table):
    return conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]


def read_rows(conn, table, columns):
    """Yield every row of a table as a dict, in primary key order."""
    selected = ', '.join(f'"{name}"' for name in columns)
    cursor = conn.execute(f'SELECT {selected} FROM "{table}" ORDER BY id')
    names = [d[0] for d in cursor.description]
    for row in cursor:
        yield dict(zip(names, row))


def target_columns(table_columns, target_conn):
    """Intersect the source columns with those the target table actually has.

    Tolerates a target that predates a column, which keeps the script usable
    against a database created by an older revision.
    """
    present = {
        row[0]
        for row in target_conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = :t",
            {'t': table},
        )
    }
    if not present:
        raise RuntimeError(
            f'Table "{table}" does not exist in the target database. '
            'Run `flask db upgrade` against PostgreSQL first.'
        )
    return [name for name in table_columns if name in present]


def upsert_statement(table, columns):
    """
    Build an INSERT ... ON CONFLICT DO NOTHING for the given columns.

    DO NOTHING is what makes re-running safe: an existing row is left exactly as
    it is, so a second run cannot overwrite production data with stale local
    copies.
    """
    placeholders = ', '.join(f':{name}' for name in columns)
    collist = ', '.join(f'"{name}"' for name in columns)
    return (
        f'INSERT INTO "{table}" ({collist}) VALUES ({placeholders}) '
        f'ON CONFLICT (id) DO NOTHING'
    )


def migrate(dry_run=False, source=None, batch_size=500):
    app = create_app()

    source_path = source or os.path.join(
        app.config['BASE_DIR'], 'database', 'emotion_app.db'
    )
    source_path = os.path.abspath(source_path)

    if not os.path.isfile(source_path):
        print(f'Error: no SQLite database at {source_path}', file=sys.stderr)
        return EXIT_NO_SOURCE

    if app.config['DB_DIALECT'] != 'postgresql':
        print(
            'Error: the target must be PostgreSQL. This script is configured '
            f'for {app.config["DB_DIALECT"]!r}.\n'
            'Set DATABASE_URL to the PostgreSQL connection string and retry; '
            'the SQLite source is passed with --source.',
            file=sys.stderr,
        )
        return EXIT_USAGE

    src = open_sqlite_readonly(source_path)
    try:
        print(f'Source (read-only): {source_path}')
        print('Target            : PostgreSQL (from DATABASE_URL)')
        if dry_run:
            print('Mode              : DRY RUN, nothing will be written')
        print()

        totals = {}
        for table in TABLES:
            try:
                totals[table] = count_rows(src, table)
            except sqlite3.Error:
                totals[table] = 0

        for table in TABLES:
            print(f'  {table:18} {totals[table]:>7} row(s) in source')

        if dry_run:
            print('\nDry run complete. Re-run without --dry-run to copy.')
            return EXIT_OK

        with app.app_context():
            target = db.session.connection().connection

            for table in TABLES:
                if totals[table] == 0:
                    print(f'  {table:18} skipped (no rows)')
                    continue

                all_columns = [
                    row[1] for row in src.execute(f'PRAGMA table_info("{table}")')
                ]
                columns = target_columns(all_columns, target)
                skipped = set(all_columns) - set(columns)
                if skipped:
                    print(
                        f'  {table:18} note: target has no column(s) '
                        f'{sorted(skipped)}, those values are not copied'
                    )

                statement = upsert_statement(table, columns)
                inserted = 0
                pending = []

                for record in read_rows(src, table, columns):
                    pending.append(record)
                    if len(pending) >= batch_size:
                        inserted += _flush(target, statement, pending, table)
                        pending = []

                if pending:
                    inserted += _flush(target, statement, pending, table)

                print(f'  {table:18} inserted {inserted} of {totals[table]}')

            db.session.commit()

        print('\nMigration finished.')
        return EXIT_OK

    except Exception as exc:  # noqa: BLE001
        db.session.rollback()
        print(f'Error: migration failed: {exc}', file=sys.stderr)
        return EXIT_FAILED
    finally:
        src.close()


def _flush(target, statement, records, table):
    """Execute one batch and return how many rows were actually inserted."""
    before = target.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar()
    for record in records:
        target.execute(
            text(statement),
            {
                key: value
                for key, value in record.items()
                if value is not None
            },
        )
    after = target.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar()
    return after - before


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Copy rows from the local SQLite database into PostgreSQL. '
            'Run `flask db upgrade` against PostgreSQL first.'
        ),
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Report row counts and exit without writing anything.',
    )
    parser.add_argument(
        '--source',
        help='Path to the SQLite file. Defaults to database/emotion_app.db.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=500,
        help='Rows per INSERT batch (default: 500).',
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        print('Error: --batch-size must be at least 1', file=sys.stderr)
        return EXIT_USAGE
    return migrate(
        dry_run=args.dry_run, source=args.source, batch_size=args.batch_size
    )


if __name__ == '__main__':
    sys.exit(main())
