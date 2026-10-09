# Emotion Detection Using Facial Expression

A Flask web application that classifies facial expressions into seven emotions
using OpenCV face detection and a Keras CNN. It runs on a local SQLite file for
development and on PostgreSQL with private Supabase Storage for production,
without changing the application code.

---

## Features

- **Accounts** — register, log in, log out; PBKDF2-SHA256 password hashing
- **Image detection** — upload a photo or capture one from the webcam
- **Live detection** — streaming webcam analysis with per-session summaries
- **Face detection** — OpenCV Haar cascade; a frame with no face is reported as
  such rather than guessed at
- **Seven-emotion classification** — Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **History** — browse, preview and delete your own past detections
- **Live session history** — duration, dominant emotion, average confidence
- **Excel export** — formatted `.xlsx` of your own history, or an admin workbook
- **Admin area** — user management, activity audit log, analytics, exports
- **Strict user isolation** — every query is scoped to the authenticated owner

---

## Architecture

The same code serves both environments. Only configuration differs.

```
                    ┌──────────────────────────────────────────┐
Browser ──HTTPS──▶  │ Flask app factory (app.py)               │
                    │  CSRF · auth · security headers · limits │
                    └───────────────┬──────────────────────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             ▼                      ▼                      ▼
      Face detection        Emotion model           Storage abstraction
      (OpenCV Haar)         (Keras CNN)             ┌──────────┬──────────┐
                                                   │  local   │ Supabase │
                                                   │ instance/│ private  │
                                                   │ uploads  │  bucket  │
                                                   └──────────┴──────────┘
                                    │
                          ┌─────────┴─────────┐
                          ▼                   ▼
                   SQLite (dev)        PostgreSQL (prod)
```

Everything is chosen by environment variables:

| Variable        | Development                | Production                          |
| --------------- | -------------------------- | ----------------------------------- |
| `DATABASE_URL`  | empty → SQLite            | PostgreSQL URI → `psycopg`           |
| `STORAGE_BACKEND` | `local`                 | `supabase` → private bucket          |

The app never silently downgrades to SQLite in production. If `DATABASE_URL` is
missing when `FLASK_ENV=production`, it refuses to start.

### ML pipeline

```
image / video frame
      │
      ├─▶ validate (extension, MIME, magic bytes, decode, size)
      │
      ├─▶ OpenCV Haar cascade → no face? report it, do not predict
      │
      ├─▶ crop, grayscale, resize to 48×48, normalise
      │
      ├─▶ Keras CNN → softmax over 7 classes
      │
      └─▶ emotion + confidence → persist → authorised preview
```

If the model cannot be loaded, every prediction route returns *Emotion model
unavailable*. The application never substitutes a random or hard-coded result.

---

## Technology stack

| Layer         | Choice                                              |
| ------------- | --------------------------------------------------- |
| Web           | Flask 3, Werkzeug, Jinja2                            |
| ORM           | SQLAlchemy 2 via Flask-SQLAlchemy                    |
| Migrations    | Alembic via Flask-Migrate                            |
| Auth          | Flask-Login, Flask-WTF (CSRF), Werkzeug security    |
| Databases     | SQLite (dev), PostgreSQL 3 / psycopg (prod)          |
| Storage       | local filesystem, Supabase Storage (private bucket) |
| Vision        | OpenCV                                              |
| Model         | TensorFlow / Keras                                  |
| Exports       | openpyxl                                            |
| Config        | python-dotenv                                       |

---

## Quick start (local, SQLite)

Requires Python 3.10+.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your configuration
copy .env.example .env         # Windows
# cp .env.example .env         # macOS / Linux

# 4. Create the schema
flask --app "app:create_app" db upgrade

# 5. Create the first admin (interactive, password is not echoed)
python scripts/create_admin.py

# 6. Run
python app.py
```

Open <http://127.0.0.1:5000>, register a normal account through the web form, and
log in.

With `DATABASE_URL` empty, `AUTO_CREATE_TABLES` defaults to on, so step 4 is
optional for a brand-new local database. It is **not** optional for an existing
one: `create_all()` never adds columns to a table that already exists, which is
exactly what `db upgrade` is for.

### Running the tests

```bash
python scripts/selftest.py
```

The suite builds a throwaway SQLite database, exercises authentication,
authorisation, CSRF, uploads, live sessions, ownership, exports and admin pages,
then deletes it. It never touches `database/emotion_app.db`.

---

## Configuration

Every setting is an environment variable. `.env` is read at startup, but real
environment variables take precedence, which is what you want on a hosting
platform. See `.env.example` for the annotated list; the ones that matter most:

| Variable | Purpose |
| --- | --- |
| `SECRET_KEY` | Signs session cookies. **Required** in production; the app refuses to start on the development fallback. |
| `DATABASE_URL` | PostgreSQL URI. Empty means local SQLite. |
| `SUPABASE_URL` | Project REST endpoint. |
| `SUPABASE_SECRET_KEY` | Server-side key for Storage (`sb_secret_...` or legacy `service_role`). |
| `STORAGE_BACKEND` | `local` or `supabase`. |
| `STORAGE_BUCKET` | Bucket name; must be private. |
| `UPLOAD_LIMIT` | Max upload size in **bytes** (default 16 MiB). |
| `LIVE_DETECTION_INTERVAL` | Seconds between accepted frames per user. |
| `EMOTION_LABELS_JSON` | Class order, if your model differs from the default. |

Generate a secret key with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### Emotion class order

The bundled `model/finalfacialemotionmodel.keras` is the checkpoint documented at
[lokeshkumar79/facial-emotion-recognition](https://huggingface.co/lokeshkumar79/facial-emotion-recognition).
Its softmax output order — and therefore `config.DEFAULT_EMOTION_LABELS` — is:

| Index | Label    |
| ----- | -------- |
| 0     | Angry    |
| 1     | Disgust  |
| 2     | Fear     |
| 3     | Happy    |
| 4     | Neutral  |
| 5     | Sad      |
| 6     | Surprise |

The saved `.keras` archive does not embed class names, so the mapping is
documented here instead of being read from the file. `EMOTION_LABELS_JSON` is an
override **only** for swapping in a checkpoint trained with a different order; it
must list the seven labels in output order. Leaving it unset uses the documented
default above. Getting the order wrong silently mislabels every prediction, so
`scripts/selftest.py` verifies the default order on every run.

### A note on `DATABASE_URL` passwords

Percent-encode reserved characters in the password. The most common failure is an
unescaped `@`:

```
postgresql://postgres.PROJECT:P@ssw0rd@aws-0-region.pooler.supabase.com:5432/postgres
```

SQLAlchemy splits on the **last** `@`, so the real host ends up inside the
password field and you get a confusing DNS error. Encode it as `%40`:

```
postgresql://postgres.PROJECT:P%40ssw0rd@aws-0-region.pooler.supabase.com:5432/postgres
```

The app detects the unescaped-`@` case and prints a warning without echoing the
value.

---

## Database migrations

Schema changes go through Alembic. Migrations live in `migrations/` and are
tracked in version control.

```bash
# Apply everything
flask --app "app:create_app" db upgrade

# After changing a model
flask --app "app:create_app" db migrate -m "describe the change"

# Confirm the database matches the models (should print nothing)
flask --app "app:create_app" db check

# Undo the last migration
flask --app "app:create_app" db downgrade -1
```

The first revision works from either starting point:

- **Empty database** — creates all four tables with foreign keys, `ON DELETE
  CASCADE`, uniqueness and indexes.
- **Pre-Alembic SQLite file** — adds the columns introduced since (`users.role`,
  `users.is_active`, `users.login_count`, `users.last_login_at`,
  `detections.processed_image_path`, `live_sessions.session_token`), creates the
  missing indexes and brings the existing constraints in line with the models.
  Existing rows are preserved; a pre-existing user picks up the real defaults
  (`role='user'`, `is_active=1`, `login_count=0`) rather than nulls.

Always back up before migrating:

```bash
copy database\emotion_app.db database\emotion_app.db.bak   # Windows
cp database/emotion_app.db database/emotion_app.db.bak     # macOS / Linux
```

---

## Supabase setup

### 1. Database

Create a Supabase project, then copy the **session mode** connection string from
**Project Settings → Database → Connection string → URI**. Put it in
`DATABASE_URL`, percent-encoding the password, then apply the schema:

```bash
flask --app "app:create_app" db upgrade
```

### 2. Private storage bucket

In **Storage**, create a bucket and **turn public access off**.

The application never serves bucket objects directly. Reads go through an
authorised route that checks the requester owns the detection, then either
redirects to a short-lived signed URL (`SIGNED_URL_TTL_SECONDS`, default 300s) or
streams the bytes. A public bucket would make every stored image readable by
anyone who guesses or leaks the key.

### 3. Keys

The server-side key needs `sb_secret_...` (new style) or the legacy
`service_role` JWT. The publishable `sb_publishable_...` / `anon` key **cannot**
manage a private bucket, so the app refuses to start if that is all it finds,
rather than failing later on the first upload.

Keep the secret key server-side only. It must never appear in client JavaScript,
in a template, or in version control.

---

## Moving local data to PostgreSQL

Optional. Once the target schema exists (`flask db upgrade` against PostgreSQL):

```bash
python scripts/migrate_sqlite_to_postgres.py --dry-run   # report only
python scripts/migrate_sqlite_to_postgres.py             # copy
```

Properties worth knowing:

- The SQLite file is opened **read-only**.
- Rows are copied with explicit primary keys and `ON CONFLICT DO NOTHING`, so
  re-running never duplicates a row and never overwrites what PostgreSQL
  already holds.
- There is no `TRUNCATE`, `DROP` or `DELETE` anywhere in the script.
- Each table is one transaction, so a failure leaves whole tables rather than
  halves.

**Stored images are not transferred.** `image_path` and `processed_image_path`
hold storage *keys*; the bytes live on local disk or in Supabase Storage. Copying
rows without the files would leave dangling references, so move the files
separately (or accept that pre-existing previews will 404 and re-upload).

---

## Admin accounts

The `admin` role is deliberately unreachable from the registration form, so it
can only be granted from the command line or by an existing admin.

```bash
python scripts/create_admin.py
python scripts/create_admin.py --username admin --email you@example.com
```

Omit `--password` and the password is read with `getpass`, so it is not echoed
and never lands in shell history. Validation is imported from `routes.auth`, so
an admin account cannot be created under weaker rules than a normal sign-up.

Exit codes: `0` success, `1` invalid input, `2` username or email already taken,
`3` database error.

---

## Security model

**Passwords** — PBKDF2-SHA256 through Werkzeug. Plain text is never stored or
logged.

**Sessions** — signed cookies, `HttpOnly`, `SameSite=Lax`, and `Secure`
automatically when `VERCEL` is set.

**CSRF** — Flask-WTF protects every state-changing form. JSON endpoints require
the token in the `X-CSRFToken` header and return a JSON error rather than an
HTML page, so the frontend can handle it. JavaScript reads the token from
`<meta name="csrf-token">`.

**Authorisation** — every detection, preview and export query is filtered by the
authenticated user id. Admin routes require the admin role. `is_active` is
checked on each request, so deactivating an account takes effect immediately
rather than at the next login.

**Live sessions** — a session is issued a 32-byte URL-safe random token, stored
server-side, and compared in constant time. A missing, empty or mismatched token
is refused; one user cannot use or end another user's session.

**Uploads** — extension, declared MIME type, magic bytes and a real decode must
all agree, plus a size cap returning HTTP 413. A file renamed to `.jpg` is
rejected. Validation happens *before* the model is consulted, so a bad upload is
refused even when the model is broken.

**Rate limiting** — a server-side sliding window on live frame submission. A
modified client cannot bypass it by changing the interval.

**Image privacy** — stored images are reachable only through an ownership-checked
route. There is no public directory serving user content.

**Security headers** — every response carries `X-Content-Type-Options: nosniff`,
`X-Frame-Options: SAMEORIGIN`, `Referrer-Policy: strict-origin-when-cross-origin`
and `Permissions-Policy: camera=(self), microphone=()`. Outside debug mode,
`Strict-Transport-Security` is added as well. A Content-Security-Policy is *not*
currently set; the templates use inline scripts and a Bootstrap CDN, so adding
one properly needs per-request nonces. Treat that as outstanding work rather
than assuming it is covered.

**No privilege escalation** — `role`, `id` and `is_active` are never read from a
submitted form. Registration always creates a `user`.

---

## Privacy

- **Images** are private user content. Uploads and camera captures are stored to
  give the user their own history, and are visible only to that user (and to an
  admin acting in an administrative capacity).
- **The audit log** (`user_activities`) records the client IP address and
  User-Agent for abuse investigation and session tracing. Both are personal
  data under GDPR. It never stores passwords, hashes, session cookies, CSRF
  tokens or API keys.
- **Exports** contain detection metadata and emotion results. Session tokens and
  password hashes are explicitly excluded from every workbook.
- **Deletion** — deleting an account removes its detections, live sessions and
  activity rows. Deleting a detection, or clearing your history, deletes the
  stored images as well as the database rows.
- **Live frames** are analysed in memory and are not persisted as images; only
  the aggregate session summary and individual detection records are stored.
- **Retention** — there is currently no automatic pruning. Rows and stored
  images accumulate until the user deletes them or the account is removed. If
  you need a retention limit, add one before processing uploads to real users.

---

## Testing

`python scripts/selftest.py` runs 29 checks against a temporary SQLite database
and prints `PASS` / `FAIL` / `BLOCKED` per check with a summary.

It covers: the documented emotion class-label order, health endpoint,
registration and login, login metrics and audit writes, CSRF on forms and JSON,
role and deactivation enforcement, admin page rendering, cross-user access
denial, upload rejection and size limits, live session token enforcement (with
inference stubbed), model-unavailable handling, export contents, deletion
confirmation and security headers.

Anything the environment prevents is reported as `BLOCKED` rather than silently
skipped or optimistically marked as passing.

---

## Deployment

### Before you deploy

- [ ] `SECRET_KEY` is a fresh random value, not the development fallback
- [ ] `DATABASE_URL` points at PostgreSQL with a percent-encoded password
- [ ] `flask db upgrade` has been applied to that database
- [ ] The Storage bucket exists and is **private**
- [ ] `SUPABASE_SECRET_KEY` is a server-side key, not the publishable key
- [ ] The model file is present and the active class order matches it — either
      the documented `DEFAULT_EMOTION_LABELS`, or an `EMOTION_LABELS_JSON` that
      is correct for your model (and not the old default order)
- [ ] `python scripts/selftest.py` passes

### A real constraint on Vercel

`tensorflow` is roughly 500–600 MB of native libraries. Vercel's serverless
function size limit is much smaller, so **a Vercel deployment that imports
TensorFlow inside the function bundle will not build**. This is a platform limit,
not something this codebase can code around.

Your options:

1. **Deploy somewhere with a normal application image** — Render, Railway,
   Fly.io, Cloud Run, or a VPS. No code changes. This is the recommended path.
2. **Split inference out** — keep the web app on Vercel and run the model behind
   a small HTTPS service; the web app calls it instead of loading Keras.
3. **Deploy to Vercel without inference.** The app handles this honestly: routes
   that need a prediction return *Emotion model unavailable*. Everything else
   (accounts, history, admin, exports) works.

Do not simply delete the `tensorflow` line to make a build pass. That yields an
app which builds and then refuses every prediction, which is far more confusing
than a build that fails loudly.

There is no `vercel.json` in this repository. Vercel's zero-configuration Flask
support detects `app.py` at the project root and uses the module-level `app`
object as the WSGI callable, so no `builds`/`routes`/`functions` overrides are
needed — and mixing `functions` with `builds` is rejected by the platform. Set
the function memory (3008 MB) and max duration (60 s) in the Vercel project
settings instead; both are at or near the plan maximum and may still be
insufficient for TensorFlow.

---

## Project layout

```
app.py                     application factory, CSRF, error handlers, WSGI app
config.py                  all configuration and environment parsing
models/
  user.py                  accounts, roles, password hashing
  detection.py             one row per detection; stores storage keys
  live_session.py          live session summaries
  user_activity.py         audit trail
routes/
  auth.py                  register, login, logout
  dashboard.py             per-user dashboard and history
  detection.py             upload, capture, live detection, authorised previews
  export.py                per-user Excel export
  admin.py                 admin area
utils/
  database.py              SQLAlchemy instance and backend selection
  storage.py               local / Supabase Storage abstraction
  security.py              access decorators, ownership checks, audit logging
  rate_limit.py            sliding-window limiter
  emotion_predictor.py     lazy model loading; fails loudly, never fakes
  excel_exporter.py        workbook generation
scripts/
  create_admin.py          admin bootstrap CLI
  migrate_sqlite_to_postgres.py   optional data copy
  selftest.py              integration test suite
migrations/                Alembic revisions (tracked in git)
templates/                 Jinja templates
static/                    CSS and JavaScript
model/                     the Keras model
database/                  local SQLite file (git-ignored)
instance/uploads/          local storage (git-ignored)
```

---

## Troubleshooting

**`Emotion model unavailable`**
The model could not be loaded. On Windows this is often an Application Control
or antivirus policy blocking a native DLL that TensorFlow depends on. The app
reports the underlying import error rather than guessing.

**`failed to resolve host '...pooler.supabase.com'`**
Almost always an unescaped `@` in the database password. See the note above.

**`DATABASE_URL is not set`**
`FLASK_ENV=production` was set without a database URL. Production never falls
back to SQLite. Either provide `DATABASE_URL` or unset `FLASK_ENV` for local
work.

**`SUPABASE storage is not usable`**
Only a publishable key was provided. A private bucket needs the server-side
secret key.

**A local detection image 404s after switching to Supabase storage**
`image_path` holds a storage key, not a file. Rows written when
`STORAGE_BACKEND=local` point at `./instance/uploads`, and a deployment on
Supabase Storage will not have those bytes. Re-upload, or copy the files across.

**`no such column: users.role`**
The database predates the current models. Run `flask --app "app:create_app" db
upgrade`.

---

## Licence and academic context

Built as an academic project. The bundled Keras model is provided as-is; the
emotion labels are not embedded in the file, so the default order is taken from
the checkpoint documentation
([lokeshkumar79/facial-emotion-recognition](https://huggingface.co/lokeshkumar79/facial-emotion-recognition))
and can be overridden with `EMOTION_LABELS_JSON` if you swap in a model trained
with a different order.



