# =============================================================================
# Emotion Detection Using Facial Expression — Configuration
# =============================================================================
# Centralized configuration for the entire application.
# All model settings, database paths, upload limits, and emotion labels
# are defined here so they can be easily modified in one place.
# =============================================================================

import os

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    """Main application configuration."""
    BASE_DIR = BASE_DIR

    # -------------------------------------------------------------------------
    # Flask Settings
    # -------------------------------------------------------------------------
    SECRET_KEY = os.environ.get('SECRET_KEY', 'emotion-detection-secret-key-change-in-production')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() in ('true', '1', 'yes')

    # -------------------------------------------------------------------------
    # Database Settings (SQLite via SQLAlchemy)
    # -------------------------------------------------------------------------
    DATABASE_DIR = os.path.join(BASE_DIR, 'database')
    DATABASE_PATH = os.path.join(DATABASE_DIR, 'emotion_app.db')
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{DATABASE_PATH}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # -------------------------------------------------------------------------
    # File Upload Settings
    # -------------------------------------------------------------------------
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

    # -------------------------------------------------------------------------
    # Emotion Detection Model Settings (Centralized)
    # -------------------------------------------------------------------------
    # Path to the pre-trained Keras model file (.h5 or .keras)
    MODEL_PATH = os.path.join(BASE_DIR, 'model', 'finalfacialemotionmodel.keras')

    # Input image dimensions expected by the model
    MODEL_INPUT_SIZE = (48, 48)

    # Color mode: 'grayscale' or 'rgb'
    # FER-2013 models typically expect grayscale (1 channel)
    MODEL_COLOR_MODE = 'grayscale'

    # Number of input channels (1 for grayscale, 3 for RGB)
    MODEL_INPUT_CHANNELS = 1

    # Normalization method: divide pixel values by this number
    # Standard normalization: pixel / 255.0 to get values in [0, 1]
    MODEL_NORMALIZATION_FACTOR = 255.0

    # -------------------------------------------------------------------------
    # Emotion Class Labels (Centralized)
    # -------------------------------------------------------------------------
    # These labels MUST match the exact order of the model's output classes.
    # For FER-2013 trained models, the standard order is:
    EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    # Number of emotion classes (derived from labels)
    NUM_EMOTION_CLASSES = len(EMOTION_LABELS)

    # -------------------------------------------------------------------------
    # Live Detection Settings
    # -------------------------------------------------------------------------
    # Interval in seconds between processing frames during live detection
    LIVE_DETECTION_INTERVAL = 2  # Process one frame every 2 seconds

    # -------------------------------------------------------------------------
    # Timestamp Display Format
    # -------------------------------------------------------------------------
    # Format: "04 September 2026, 10:24:15 AM"
    TIMESTAMP_DISPLAY_FORMAT = '%d %B %Y, %I:%M:%S %p'

    # Format for date only: "04 Sep 2026"
    DATE_DISPLAY_FORMAT = '%d %b %Y'

    # Format for time only: "10:24:15 AM"
    TIME_DISPLAY_FORMAT = '%I:%M:%S %p'

    # -------------------------------------------------------------------------
    # Face Detection Settings (OpenCV Haar Cascade)
    # -------------------------------------------------------------------------
    FACE_DETECTION_SCALE_FACTOR = 1.3
    FACE_DETECTION_MIN_NEIGHBORS = 5
    FACE_DETECTION_MIN_SIZE = (30, 30)
