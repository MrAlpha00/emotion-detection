// =============================================================================
// Camera Capture JavaScript
// =============================================================================
// Handles browser camera access for the face detection page.
// Uses navigator.mediaDevices.getUserMedia() for camera stream.
// Captures frames to canvas, converts to base64, and sends to server.
// =============================================================================

let cameraStream = null;

/**
 * Start the browser camera and show the video feed.
 */
function startCamera() {
    const video = document.getElementById('camera-video');
    const feedDiv = document.getElementById('camera-feed');
    const promptDiv = document.getElementById('camera-start-prompt');
    const capturedDiv = document.getElementById('captured-preview');

    // Request camera access
    navigator.mediaDevices.getUserMedia({
        video: {
            facingMode: 'user',
            width: { ideal: 640 },
            height: { ideal: 480 }
        },
        audio: false
    })
    .then(function(stream) {
        cameraStream = stream;
        video.srcObject = stream;

        // Show camera feed, hide prompt
        feedDiv.style.display = 'block';
        promptDiv.style.display = 'none';
        capturedDiv.style.display = 'none';
    })
    .catch(function(error) {
        console.error('Camera access error:', error);
        let message = 'Unable to access camera. ';
        if (error.name === 'NotAllowedError') {
            message += 'Please allow camera access in your browser settings.';
        } else if (error.name === 'NotFoundError') {
            message += 'No camera device found.';
        } else {
            message += error.message;
        }
        alert(message);
    });
}

/**
 * Stop the camera stream.
 */
function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(function(track) {
            track.stop();
        });
        cameraStream = null;
    }

    const video = document.getElementById('camera-video');
    video.srcObject = null;

    document.getElementById('camera-feed').style.display = 'none';
    document.getElementById('camera-start-prompt').style.display = 'block';
    document.getElementById('captured-preview').style.display = 'none';
}

/**
 * Capture the current video frame to a canvas.
 */
function captureFrame() {
    const video = document.getElementById('camera-video');
    const canvas = document.getElementById('capture-canvas');
    const ctx = canvas.getContext('2d');

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw the current video frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Show captured preview, hide camera feed
    document.getElementById('camera-feed').style.display = 'none';
    document.getElementById('captured-preview').style.display = 'block';

    // Stop the camera stream since we have the frame
    if (cameraStream) {
        cameraStream.getTracks().forEach(function(track) {
            track.stop();
        });
        cameraStream = null;
    }
}

/**
 * Retake — go back to camera mode.
 */
function retakePhoto() {
    document.getElementById('captured-preview').style.display = 'none';
    startCamera();
}

/**
 * Send the captured image to the server for emotion analysis.
 */
function analyzeCapturedImage() {
    const canvas = document.getElementById('capture-canvas');
    const imageData = canvas.toDataURL('image/jpeg', 0.9);
    const overlay = document.getElementById('loading-overlay');
    const analyzeBtn = document.getElementById('btn-analyze-capture');

    // Show loading state
    if (overlay) overlay.style.display = 'flex';
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Analyzing...';

    fetch('/detect/capture', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: imageData })
    })
    .then(function(response) {
        return response.json();
    })
    .then(function(data) {
        if (overlay) overlay.style.display = 'none';

        if (data.success && data.redirect) {
            // Redirect to result page
            window.location.href = data.redirect;
        } else {
            // Show error
            alert(data.error || 'An error occurred during analysis.');
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="bi bi-search me-1"></i>Analyze Emotion';
        }
    })
    .catch(function(error) {
        if (overlay) overlay.style.display = 'none';
        console.error('Analysis error:', error);
        alert('Failed to analyze the image. Please try again.');
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="bi bi-search me-1"></i>Analyze Emotion';
    });
}
