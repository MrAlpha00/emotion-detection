# =============================================================================
# Emotion Predictor
# =============================================================================
# Loads and validates the Keras emotion detection model at startup.
# Provides prediction functionality with strict validation.
# NO fallback/demo/random prediction mode — if the model is unavailable,
# all prediction requests are refused with a clear error.
# =============================================================================

import os
import numpy as np
from config import Config


class ModelUnavailableError(Exception):
    """Raised when a prediction is attempted but no valid model is loaded."""
    pass


class EmotionPredictor:
    """
    Singleton emotion predictor that loads a Keras model once at startup.
    Validates model input/output dimensions against the configuration.
    Refuses to predict if the model is missing, corrupt, or incompatible.
    """

    def __init__(self):
        self._model = None
        self._available = False
        self._status_message = 'Model not yet loaded.'

    def load_model(self):
        """
        Attempt to load and validate the emotion detection model.
        Sets self._available = True only if the model passes all checks.
        """
        model_path = Config.MODEL_PATH

        # --- Check 1: Does the model file exist? ---
        if not os.path.exists(model_path):
            self._available = False
            self._status_message = (
                f'Emotion model file not found. '
                f'Please place your trained model at: {model_path}'
            )
            print(f"[EmotionPredictor] {self._status_message}")
            return

        # --- Check 2: Can we load the model? ---
        try:
            # Import TensorFlow/Keras only when needed (heavy import)
            import tensorflow as tf
            # Suppress TF info logs during loading
            os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

            self._model = tf.keras.models.load_model(model_path, compile=False)
            print(f"[EmotionPredictor] Model file loaded from: {model_path}")
        except Exception as e:
            self._model = None
            self._available = False
            self._status_message = (
                f'Failed to load emotion model: {str(e)}. '
                f'The model file may be corrupt or incompatible.'
            )
            print(f"[EmotionPredictor] {self._status_message}")
            return

        # --- Check 3: Validate the model ---
        self._validate_model()

    def _validate_model(self):
        """
        Validate the loaded model's input and output shapes against config.
        Expected input:  (None, 48, 48, 1) for 48x48 grayscale
        Expected output: (None, 7) for 7 emotion classes
        """
        if self._model is None:
            self._available = False
            self._status_message = 'No model loaded to validate.'
            return

        try:
            # Get model input shape
            input_shape = self._model.input_shape
            expected_h, expected_w = Config.MODEL_INPUT_SIZE
            expected_channels = Config.MODEL_INPUT_CHANNELS

            # input_shape is typically (None, height, width, channels)
            if len(input_shape) == 4:
                _, h, w, c = input_shape
            elif len(input_shape) == 3:
                # Some models may have shape (None, height*width) — flat input
                self._model = None
                self._available = False
                self._status_message = (
                    f'Model input shape {input_shape} is not a valid image shape. '
                    f'Expected (None, {expected_h}, {expected_w}, {expected_channels}).'
                )
                print(f"[EmotionPredictor] {self._status_message}")
                return
            else:
                self._model = None
                self._available = False
                self._status_message = (
                    f'Unexpected model input shape: {input_shape}. '
                    f'Expected (None, {expected_h}, {expected_w}, {expected_channels}).'
                )
                print(f"[EmotionPredictor] {self._status_message}")
                return

            # Validate input dimensions
            if h != expected_h or w != expected_w:
                self._model = None
                self._available = False
                self._status_message = (
                    f'Model input size mismatch: model expects {h}x{w}, '
                    f'but config expects {expected_h}x{expected_w}. '
                    f'Update MODEL_INPUT_SIZE in config.py to match your model.'
                )
                print(f"[EmotionPredictor] {self._status_message}")
                return

            # Validate input channels (grayscale vs RGB)
            if c != expected_channels:
                self._model = None
                self._available = False
                self._status_message = (
                    f'Model color mode mismatch: model expects {c} channel(s), '
                    f'but config expects {expected_channels} '
                    f'({"grayscale" if expected_channels == 1 else "RGB"}). '
                    f'Update MODEL_COLOR_MODE and MODEL_INPUT_CHANNELS in config.py.'
                )
                print(f"[EmotionPredictor] {self._status_message}")
                return

            # Get model output shape
            output_shape = self._model.output_shape
            # output_shape is typically (None, num_classes)
            if len(output_shape) == 2:
                _, num_classes = output_shape
            else:
                self._model = None
                self._available = False
                self._status_message = (
                    f'Unexpected model output shape: {output_shape}. '
                    f'Expected (None, {Config.NUM_EMOTION_CLASSES}).'
                )
                print(f"[EmotionPredictor] {self._status_message}")
                return

            # Validate number of output classes
            if num_classes != Config.NUM_EMOTION_CLASSES:
                self._model = None
                self._available = False
                self._status_message = (
                    f'Model output class count mismatch: model has {num_classes} classes, '
                    f'but config defines {Config.NUM_EMOTION_CLASSES} emotion labels '
                    f'{Config.EMOTION_LABELS}. '
                    f'Update EMOTION_LABELS in config.py to match your model.'
                )
                print(f"[EmotionPredictor] {self._status_message}")
                return

            # All checks passed
            self._available = True
            self._status_message = (
                f'Model loaded successfully. '
                f'Input: {expected_h}x{expected_w}x{expected_channels} '
                f'({"grayscale" if expected_channels == 1 else "RGB"}), '
                f'Output: {num_classes} classes {Config.EMOTION_LABELS}.'
            )
            print(f"[EmotionPredictor] {self._status_message}")

        except Exception as e:
            self._model = None
            self._available = False
            self._status_message = f'Model validation failed: {str(e)}'
            print(f"[EmotionPredictor] {self._status_message}")

    def is_available(self):
        """Returns True if a valid, validated model is loaded and ready."""
        return self._available

    def get_status_message(self):
        """Returns a human-readable status message about the model."""
        return self._status_message

    def preprocess(self, face_image):
        """
        Preprocess a face image for the emotion model.

        Args:
            face_image: OpenCV image (numpy array) of the cropped face.

        Returns:
            Preprocessed numpy array ready for model.predict()
        """
        import cv2

        h, w = Config.MODEL_INPUT_SIZE
        channels = Config.MODEL_INPUT_CHANNELS

        # Convert to grayscale if needed
        if channels == 1 and len(face_image.shape) == 3:
            processed = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        elif channels == 3 and len(face_image.shape) == 2:
            processed = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
        else:
            processed = face_image.copy()

        # Resize to expected input dimensions
        processed = cv2.resize(processed, (w, h))

        # Normalize pixel values
        processed = processed.astype('float32') / Config.MODEL_NORMALIZATION_FACTOR

        # Reshape for model input: (1, height, width, channels)
        if channels == 1:
            processed = processed.reshape(1, h, w, 1)
        else:
            processed = processed.reshape(1, h, w, 3)

        return processed

    def predict(self, face_image):
        """
        Predict the emotion from a preprocessed face image.

        Args:
            face_image: OpenCV image (numpy array) of the cropped face.

        Returns:
            dict with keys:
                - 'emotion': str, the predicted emotion label
                - 'confidence': float, confidence percentage (e.g., 94.27)
                - 'probabilities': dict mapping each emotion label to its
                  probability percentage

        Raises:
            ModelUnavailableError: If no valid model is loaded.
        """
        if not self._available or self._model is None:
            raise ModelUnavailableError(
                'Emotion model is not available. '
                'Please place a valid model file at: ' + Config.MODEL_PATH
            )

        # Preprocess the face image
        processed = self.preprocess(face_image)

        # Run prediction through the model
        predictions = self._model.predict(processed, verbose=0)

        # predictions is an array of shape (1, num_classes)
        probabilities = predictions[0]

        # Find the class with the highest probability
        predicted_index = int(np.argmax(probabilities))
        predicted_emotion = Config.EMOTION_LABELS[predicted_index]
        confidence = float(probabilities[predicted_index]) * 100

        # Build probability dictionary for all classes
        prob_dict = {}
        for i, label in enumerate(Config.EMOTION_LABELS):
            prob_dict[label] = round(float(probabilities[i]) * 100, 2)

        return {
            'emotion': predicted_emotion,
            'confidence': round(confidence, 2),
            'probabilities': prob_dict
        }


# Global singleton instance — used throughout the application
emotion_predictor = EmotionPredictor()
