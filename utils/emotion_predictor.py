# =============================================================================
# Emotion Predictor
# =============================================================================
# Loads and validates the Keras emotion model, then performs inference.
#
# Design notes
# ------------
# * There is NO fallback, demo or random prediction path. If the model is
#   missing, corrupt or incompatible, every prediction attempt raises
#   ModelUnavailableError and the UI shows "Emotion model unavailable".
#   Fake predictions are never generated or stored.
# * The model is loaded lazily on first use and cached in a module-level
#   singleton. On a Vercel serverless instance this means the (expensive) load
#   happens on the first request of a cold start and is then reused for the
#   lifetime of that warm instance. Nothing about the cache is treated as
#   persistent state - the database and object storage remain the source of
#   truth.
# * The class order is defined once in config.EMOTION_LABELS and is validated
#   against the model's real output dimension at load time. Getting the order
#   wrong would silently mislabel every prediction, so the order is documented
#   explicitly rather than assumed.
# =============================================================================

import logging
import os
import threading

from config import Config

logger = logging.getLogger(__name__)

# Keep TensorFlow quiet about CPU feature detection.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')


class ModelUnavailableError(Exception):
    """Raised when a prediction is attempted but no valid model is loaded."""


class EmotionPredictor:
    """
    Thread-safe, lazily initialised wrapper around the Keras model.
    """

    def __init__(self):
        self._model = None
        self._available = False
        self._status_message = 'Model not yet loaded.'
        self._lock = threading.Lock()
        # Fingerprint of the last *failed* attempt. Retrying an import that has
        # already failed leaves the broken module half-initialised in
        # sys.modules, so the retry must only happen when the model file itself
        # has actually changed.
        self._failure_fingerprint = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _model_fingerprint(self):
        """
        Identify the model file on disk.

        Any change to path, modification time or size produces a new
        fingerprint, which re-enables the lazy retry.
        """
        path = Config.MODEL_PATH
        if not path:
            return ('no-path',)
        try:
            stat = os.stat(path)
        except OSError:
            return ('missing', str(path))
        return (str(path), stat.st_mtime_ns, stat.st_size)

    def ensure_loaded(self):
        """
        Load and validate the model if that has not happened yet.

        Safe to call on every request: after a successful load this is a single
        boolean check, and after a failure whose cause is unchanged it is a
        single tuple comparison.
        """
        if self._available:
            return True

        fingerprint = self._model_fingerprint()
        if self._failure_fingerprint == fingerprint:
            # Same broken model as last time. Do not re-enter the import path:
            # a failed native import can recurse until the stack overflows.
            return False

        return self.load_model()

    def load_model(self):
        """
        Load and fully validate the emotion model.

        Sets ``self._available`` only when every check passes.
        """
        with self._lock:
            if self._available:
                return True

            self._failure_fingerprint = None
            self._model = None
            self._available = False

            model_path = Config.MODEL_PATH
            fingerprint = self._model_fingerprint()

            # --- Check 1: does the file exist? ---
            if not model_path or not os.path.exists(model_path):
                self._status_message = (
                    'Emotion model unavailable: model file not found at '
                    f'{model_path}. Place the trained model there or set MODEL_PATH.'
                )
                logger.warning(self._status_message)
                self._failure_fingerprint = fingerprint
                return False

            # --- Check 2: can TensorFlow/Keras load it? ---
            try:
                import tensorflow as tf  # noqa: PLC0415 - deliberately lazy

                self._model = tf.keras.models.load_model(model_path, compile=False)
                logger.info('Emotion model file loaded from %s', model_path)
            except Exception as exc:
                self._model = None
                self._status_message = (
                    f'Emotion model unavailable: failed to load the model file ({exc}). '
                    'It may be corrupt or built with an incompatible Keras version.'
                )
                logger.warning(self._status_message)
                self._failure_fingerprint = fingerprint
                return False

            # --- Check 3: validate shapes and class mapping ---
            return self._validate_model(fingerprint=fingerprint)

    def _validate_model(self, fingerprint=None):
        """
        Confirm the model matches the configured preprocessing contract.

        Validates input height/width/channels, output rank, and - crucially -
        that the number of output classes equals len(Config.EMOTION_LABELS).
        """
        if self._model is None:
            self._available = False
            self._status_message = 'Emotion model unavailable: no model loaded.'
            self._failure_fingerprint = fingerprint
            return False

        expected_h, expected_w = Config.MODEL_INPUT_SIZE
        expected_channels = 1 if Config.MODEL_COLOR_MODE == 'grayscale' else 3

        try:
            input_shape = getattr(self._model, 'input_shape', None)
            if isinstance(input_shape, list):
                # Multi-input model: not supported by this project.
                self._fail(
                    f'Model has {len(input_shape)} input tensors. This application '
                    'expects a single-image model.'
                )
                return False

            if not input_shape or len(input_shape) != 4:
                self._fail(
                    f'Model input shape {input_shape} is not a 4D image tensor. '
                    f'Expected (None, {expected_h}, {expected_w}, {expected_channels}).'
                )
                return False

            _, h, w, c = input_shape
            if None in (h, w, c):
                self._fail('Model input shape is only partially defined.')
                return False
            if int(h) != expected_h or int(w) != expected_w:
                self._fail(
                    f'Model input size mismatch: model expects {h}x{w}, config expects '
                    f'{expected_h}x{expected_w}. Update MODEL_INPUT_SIZE in config.py.'
                )
                return False
            if int(c) != expected_channels:
                self._fail(
                    f'Model channel mismatch: model expects {c} channel(s), config expects '
                    f'{expected_channels} ({Config.MODEL_COLOR_MODE}). '
                    'Update MODEL_COLOR_MODE in config.py.'
                )
                return False

            output_shape = getattr(self._model, 'output_shape', None)
            if isinstance(output_shape, list):
                output_shape = output_shape[0]
            if not output_shape or len(output_shape) != 2:
                self._fail(
                    f'Model output shape {output_shape} is unexpected. Expected a single '
                    f'(None, {Config.NUM_EMOTION_CLASSES}) vector.'
                )
                return False

            _, num_classes = output_shape
            if num_classes != Config.NUM_EMOTION_CLASSES:
                self._fail(
                    f'Model output class count mismatch: model emits {num_classes} classes but '
                    f'config defines {Config.NUM_EMOTION_CLASSES} labels '
                    f'{Config.EMOTION_LABELS}. Refusing to predict, because a wrong class '
                    'order would silently mislabel every result.'
                )
                return False

            # Every check passed.
            self._available = True
            self._status_message = (
                f'Model ready: input {h}x{w}x{c} ({Config.MODEL_COLOR_MODE}), '
                f'output {num_classes} classes mapped to {Config.EMOTION_LABELS}.'
            )
            logger.info(self._status_message)
            return True

        except Exception as exc:
            self._fail(f'Model validation error: {exc}')
            return False

    def _fail(self, message, fingerprint=None):
        """Reject the model. ``self._model`` is dropped so nothing can use it."""
        self._model = None
        self._available = False
        self._status_message = message
        self._failure_fingerprint = fingerprint if fingerprint is not None else self._model_fingerprint()
        logger.warning(message)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def is_available(self):
        """True when a validated model is loaded and ready."""
        return self._available

    def get_status_message(self):
        """Human-readable status, safe to show in the UI and log."""
        return self._status_message

    @property
    def emotion_labels(self):
        """The class order this model is being interpreted with."""
        return list(Config.EMOTION_LABELS)

    def warm_up(self):
        """
        Best-effort load at startup.

        Never raises. A missing or broken model must not stop the process from
        booting - the UI simply reports "Emotion model unavailable" and all
        detection routes refuse to run.
        """
        try:
            return self.ensure_loaded()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning('Model warm-up failed: %s', exc)
            self._available = False
            return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def preprocess(self, face_image):
        """
        Convert a cropped BGR face into the tensor the model expects.

        grayscale conversion -> resize to MODEL_INPUT_SIZE -> scale to [0, 1]
        -> reshape to (1, H, W, C)
        """
        import cv2
        import numpy as np  # noqa: F401 - used for typing/shape only

        if face_image is None or getattr(face_image, 'size', 0) == 0:
            raise ModelUnavailableError('The supplied face image is empty.')

        h, w = Config.MODEL_INPUT_SIZE
        channels = expected_channels = 1 if Config.MODEL_COLOR_MODE == 'grayscale' else 3

        # Colour conversion
        if channels == 1:
            if len(face_image.shape) == 3:
                processed = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                processed = face_image.copy()
        else:
            if len(face_image.shape) == 2:
                processed = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
            else:
                processed = face_image.copy()

        # Resize
        processed = cv2.resize(processed, (w, h), interpolation=cv2.INTER_AREA)

        # Normalise to [0, 1]
        processed = processed.astype('float32') / Config.MODEL_NORMALIZATION_FACTOR

        # (1, H, W, C)
        processed = processed.reshape(1, h, w, expected_channels)
        return processed

    def predict(self, face_image):
        """
        Classify a cropped face.

        Returns:
            dict:
              emotion        str    highest-probability label
              confidence     float  percentage, e.g. 94.27
              probabilities  dict   label -> percentage for every class

        Raises:
            ModelUnavailableError: when no validated model is available.
        """
        if not self.ensure_loaded() or self._model is None:
            raise ModelUnavailableError(self.get_status_message())

        processed = self.preprocess(face_image)

        try:
            predictions = self._model.predict(processed, verbose=0)
        except Exception as exc:
            raise ModelUnavailableError(f'Prediction failed: {exc}') from exc

        if predictions is None or len(predictions) == 0:
            raise ModelUnavailableError('The model returned an empty result.')

        probabilities = predictions[0]

        # Guard against a model whose output rank differs from what we validated.
        if len(probabilities) != Config.NUM_EMOTION_CLASSES:
            self._fail(
                f'Model returned {len(probabilities)} values but '
                f'{Config.NUM_EMOTION_CLASSES} labels are configured. Refusing to map them.'
            )
            raise ModelUnavailableError(self.get_status_message())

        predicted_index = int(probabilities.argmax())
        predicted_emotion = Config.EMOTION_LABELS[predicted_index]
        confidence = float(probabilities[predicted_index]) * 100.0

        prob_dict = {
            label: round(float(probabilities[i]) * 100.0, 2)
            for i, label in enumerate(Config.EMOTION_LABELS)
        }

        return {
            'emotion': predicted_emotion,
            'confidence': round(confidence, 2),
            'probabilities': prob_dict,
        }


# Global singleton used throughout the application.
emotion_predictor = EmotionPredictor()
