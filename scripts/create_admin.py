#!/usr/bin/env python3
# =============================================================================
# Admin Creation Script
# =============================================================================
# Creates an admin account from the command line.
#
# Admin is deliberately NOT reachable from the web registration form, so this
# script (or an existing signed-in admin) is the only way to obtain the role.
#
# Usage:
#     python scripts/create_admin.py
#     python scripts/create_admin.py --username admin --email a@b.com
#     python scripts/create_admin.py --username admin --email a@b.com \
#         --password 'correct horse battery staple'
#
# Without --password the password is read interactively with getpass, so it is
# never echoed to the terminal and never ends up in shell history or in a
# process listing.
#
# Validation is imported from routes.auth rather than reimplemented, so an admin
# account can never be created with weaker rules than a normal registration.
# =============================================================================

import argparse
import getpass
import os
import sys

# Make the project root importable when run as `python scripts/create_admin.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app  # noqa: E402
from models.user import User  # noqa: E402
from routes.auth import (  # noqa: E402
    MIN_PASSWORD_LENGTH,
    valid_email,
    valid_username,
)
from utils.database import db  # noqa: E402

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_CONFLICT = 2
EXIT_DB_ERROR = 3


def prompt_for_password():
    """Read a password without echoing it, and confirm it."""
    password = getpass.getpass('Admin password: ')
    if not password:
        return None, 'Password must not be empty.'
    confirm = getpass.getpass('Confirm password: ')
    if password != confirm:
        return None, 'Passwords do not match.'
    return password, None


def collect_details(args):
    """
    Resolve username, email and password from arguments or interactively.

    Returns (details, error_message). ``details`` is None when validation fails.
    """
    username = (args.username or input('Admin username: ')).strip()
    email = (args.email or input('Admin email: ')).strip().lower()

    if not username:
        return None, 'Username is required.'
    if not valid_username(username):
        return None, (
            'Username must be 3-80 characters using letters, numbers, '
            'underscores or hyphens.'
        )

    if not email:
        return None, 'Email is required.'
    if not valid_email(email):
        return None, 'Please enter a valid email address.'

    if args.password is not None:
        password = args.password
    else:
        password, error = prompt_for_password()
        if error:
            return None, error

    if len(password) < MIN_PASSWORD_LENGTH:
        return None, (
            f'Password must be at least {MIN_PASSWORD_LENGTH} characters long.'
        )

    return {'username': username, 'email': email, 'password': password}, None


def create_admin(app, details):
    """
    Insert the admin account.

    Returns (exit_code, message). The caller decides how to print it.
    """
    with app.app_context():
        clash = User.query.filter(
            (User.username == details['username'])
            | (User.email == details['email'])
        ).first()
        if clash is not None:
            field = 'username' if clash.username == details['username'] else 'email'
            return EXIT_CONFLICT, (
                f'Error: that {field} is already registered to '
                f'"{clash.username}". Pick a different one.'
            )

        admin = User(
            username=details['username'],
            email=details['email'],
            role='admin',
            is_active=True,
        )
        admin.set_password(details['password'])

        db.session.add(admin)
        try:
            db.session.commit()
        except Exception as exc:  # noqa: BLE001
            db.session.rollback()
            return EXIT_DB_ERROR, f'Error: could not save the account ({exc}).'

        return EXIT_OK, (
            f'Success: admin "{admin.username}" <{admin.email}> created.\n'
            'Log in at /login. Consider promoting further admins from '
            '/admin/users rather than re-running this script.'
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description='Create an admin account for the emotion detection app.',
    )
    parser.add_argument(
        '--username', help='Admin username (prompted for when omitted).'
    )
    parser.add_argument(
        '--email', help='Admin email address (prompted for when omitted).'
    )
    parser.add_argument(
        '--password',
        help=(
            'Admin password. Omit to be prompted securely with getpass. '
            'Supplying it on the command line exposes it in shell history and '
            'in the process list, so prefer the prompt.'
        ),
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    try:
        details, error = collect_details(args)
    except (EOFError, KeyboardInterrupt):
        print('\nCancelled.', file=sys.stderr)
        return EXIT_INVALID

    if error:
        print(f'Error: {error}', file=sys.stderr)
        return EXIT_INVALID

    try:
        app = create_app()
    except Exception as exc:  # noqa: BLE001
        print(f'Error: the application could not start ({exc}).', file=sys.stderr)
        return EXIT_DB_ERROR

    code, message = create_admin(app, details)
    stream = sys.stdout if code == EXIT_OK else sys.stderr
    print(message, file=stream)
    return code


if __name__ == '__main__':
    sys.exit(main())
