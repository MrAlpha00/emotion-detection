# =============================================================================
# Face Detector
# =============================================================================
# Uses OpenCV's Haar Cascade classifier for face detection.
# Provides functions to detect faces, crop them, and draw bounding boxes.
# Haar Cascade is chosen for simplicity and academic explainability.
# =============================================================================

import cv2
import numpy as np
from config import Config


# Load the Haar Cascade classifier once at module level
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def detect_faces(image):
    """
    Detect faces in an image using Haar Cascade.

    Args:
        image: OpenCV image (numpy array, BGR or grayscale).

    Returns:
        List of face bounding boxes as (x, y, w, h) tuples.
        Empty list if no faces found or image is invalid.
    """
    if image is None or image.size == 0:
        return []

    # Convert to grayscale for detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Detect faces with configurable parameters
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=Config.FACE_DETECTION_SCALE_FACTOR,
        minNeighbors=Config.FACE_DETECTION_MIN_NEIGHBORS,
        minSize=Config.FACE_DETECTION_MIN_SIZE
    )

    # Convert from numpy array to list of tuples
    if isinstance(faces, np.ndarray) and len(faces) > 0:
        return [tuple(face) for face in faces]
    return []


def crop_face(image, bbox, padding=20):
    """
    Crop a detected face from the image with optional padding.

    Args:
        image: OpenCV image (numpy array).
        bbox: Bounding box as (x, y, w, h).
        padding: Extra pixels around the face for context.

    Returns:
        Cropped face image (numpy array), or None if invalid.
    """
    if image is None or bbox is None:
        return None

    x, y, w, h = bbox
    img_h, img_w = image.shape[:2]

    # Add padding while staying within image bounds
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)

    cropped = image[y1:y2, x1:x2]

    if cropped.size == 0:
        return None

    return cropped


def get_largest_face(faces):
    """
    From a list of detected faces, return the largest one (by area).
    This is used when multiple faces are detected to pick the primary face.

    Args:
        faces: List of (x, y, w, h) bounding boxes.

    Returns:
        The (x, y, w, h) of the largest face, or None if list is empty.
    """
    if not faces:
        return None

    # Sort by area (w * h) descending, return the largest
    return max(faces, key=lambda f: f[2] * f[3])


def draw_bounding_boxes(image, faces, color=(0, 255, 0), thickness=2):
    """
    Draw rectangles around detected faces on a copy of the image.

    Args:
        image: OpenCV image (numpy array).
        faces: List of (x, y, w, h) bounding boxes.
        color: BGR color tuple for the rectangles.
        thickness: Line thickness in pixels.

    Returns:
        A copy of the image with bounding boxes drawn.
    """
    if image is None:
        return image

    result = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        # Add a small label background
        cv2.rectangle(result, (x, y - 20), (x + w, y), color, -1)
        cv2.putText(result, 'Face', (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return result


def draw_emotion_boxes(image, faces, emotions, confidences, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes with emotion labels and confidence scores.

    Args:
        image: OpenCV image (numpy array).
        faces: List of (x, y, w, h) bounding boxes.
        emotions: List of emotion labels corresponding to each face.
        confidences: List of confidence scores corresponding to each face.
        color: BGR color tuple.
        thickness: Line thickness.

    Returns:
        A copy of the image with annotated bounding boxes.
    """
    if image is None:
        return image

    result = image.copy()
    for i, (x, y, w, h) in enumerate(faces):
        emotion = emotions[i] if i < len(emotions) else 'Unknown'
        confidence = confidences[i] if i < len(confidences) else 0.0

        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)

        label = f'{emotion}: {confidence:.1f}%'
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(result, (x, y - label_size[1] - 10),
                      (x + label_size[0] + 10, y), color, -1)
        cv2.putText(result, label, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return result
