# =============================================================================
# Timezone Helpers
# =============================================================================
# Single source of truth for how timestamps move between storage, display and
# export.
#
# STORAGE (UTC)
#   Every timestamp column is a naive db.DateTime written from a timezone-aware
#   datetime.now(timezone.utc) value (models/user.py, models/detection.py,
#   models/live_session.py, models/user_activity.py). A naive value read back
#   from the database is therefore ALWAYS interpreted as UTC here - never as
#   server-local time.
#
# DISPLAY (Asia/Kolkata by default)
#   Everything rendered in a template or written into an Excel export goes
#   through to_display(), which converts the UTC instant to the IANA zone
#   configured by Config.DISPLAY_TIMEZONE using zoneinfo. The conversion is a
#   real timezone conversion (DST-aware), not a hardcoded "+5:30" offset.
#
# The module reads Config lazily from the config module because
# app.create_app() swaps config.Config for the selected config class; binding
# the class at import time could capture the wrong one.
# =============================================================================

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import config as config_module

DEFAULT_DISPLAY_TIMEZONE = 'Asia/Kolkata'
_UTC = timezone.utc


def display_timezone_name():
    """IANA name of the zone displayed timestamps are converted to."""
    return (
        getattr(config_module.Config, 'DISPLAY_TIMEZONE', None)
        or DEFAULT_DISPLAY_TIMEZONE
    )


def display_timezone():
    """
    Resolve the configured display zone.

    Falls back to Asia/Kolkata and finally to UTC if the configured name is
    missing or not a valid IANA zone, so a typo can never break rendering.
    """
    try:
        return ZoneInfo(display_timezone_name())
    except Exception:
        try:
            return ZoneInfo(DEFAULT_DISPLAY_TIMEZONE)
        except Exception:
            return _UTC


def as_utc(value):
    """
    Return ``value`` as a timezone-aware UTC datetime.

    Naive datetimes are assumed to be UTC (matching how the application writes
    them). Non-datetime values are returned unchanged.
    """
    if not isinstance(value, datetime):
        return value
    if value.tzinfo is None:
        return value.replace(tzinfo=_UTC)
    return value.astimezone(_UTC)


def to_display(value):
    """
    Convert a stored timestamp to the display timezone.

    Accepts:
      - naive datetime      -> assumed UTC, converted to the display zone
      - aware datetime      -> converted to the display zone
      - ISO-8601 string     -> parsed, then converted (e.g. a session payload)
      - None / other types  -> returned unchanged (None stays None)

    The returned datetime is timezone-aware and carries the display zone, so
    calling .strftime() on it shows local (IST) wall-clock time.
    """
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        if text.endswith(('Z', 'z')):
            text = text[:-1] + '+00:00'
        try:
            value = datetime.fromisoformat(text)
        except ValueError:
            # Not a timestamp string - leave it alone rather than break output.
            return value

    if not isinstance(value, datetime):
        return value

    if value.tzinfo is None:
        value = value.replace(tzinfo=_UTC)
    return value.astimezone(display_timezone())


def format_display(value, fmt):
    """Convert to the display timezone and format it; tolerant of None."""
    converted = to_display(value)
    if converted is None or isinstance(converted, str):
        return '' if converted is None else converted
    try:
        return converted.strftime(fmt)
    except (ValueError, OSError):
        return converted.isoformat()
