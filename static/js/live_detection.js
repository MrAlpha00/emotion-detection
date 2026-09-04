// =============================================================================
// Live Detection JavaScript
// =============================================================================
// Handles live webcam emotion detection with interval-based frame processing.
// Manages session lifecycle (start/stop), draws face bounding boxes on canvas
// overlay, updates emotion display, session timer, and emotion distribution.
// =============================================================================

let liveStream = null;
let liveInterval = null;
let sessionTimer = null;
let sessionStartTime = null;
let sessionId = null;
let dbSessionId = null;
let emotionCounts = {};
let totalLiveDetections = 0;
let isProcessing = false;

// Detection interval in milliseconds (matches config.LIVE_DETECTION_INTERVAL)
const DETECTION_INTERVAL_MS = 2000;

/**
 * Start live emotion detection session.
 */
function startLiveDetection() {
    const video = document.getElementById('live-video');
    const startBtn = document.getElementById('btn-start-live');
    const stopBtn = document.getElementById('btn-stop-live');

    startBtn.disabled = true;
    startBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Starting...';

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
        liveStream = stream;
        video.srcObject = stream;

        // Wait for video to be ready
        video.onloadedmetadata = function() {
            // Set overlay canvas size
            const overlay = document.getElementById('live-overlay');
            overlay.width = video.videoWidth;
            overlay.height = video.videoHeight;

            // Start server session
            startServerSession();
        };
    })
    .catch(function(error) {
        console.error('Camera error:', error);
        let message = 'Unable to access camera. ';
        if (error.name === 'NotAllowedError') {
            message += 'Please allow camera access in your browser settings.';
        } else if (error.name === 'NotFoundError') {
            message += 'No camera device found.';
        } else {
            message += error.message;
        }
        alert(message);
        startBtn.disabled = false;
        startBtn.innerHTML = '<i class="bi bi-play-fill me-1"></i>Start Detection';
    });
}

/**
 * Create a session on the server.
 */
function startServerSession() {
    fetch('/live/start-session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    })
    .then(r => r.json())
    .then(function(data) {
        if (data.success) {
            sessionId = data.session_id;
            dbSessionId = data.db_session_id;
            sessionStartTime = new Date();

            // Reset counters
            emotionCounts = {};
            totalLiveDetections = 0;

            // Show/hide buttons
            document.getElementById('btn-start-live').style.display = 'none';
            document.getElementById('btn-stop-live').style.display = 'inline-block';
            document.getElementById('live-recording-badge').style.display = 'inline';
            document.getElementById('session-summary-card').style.display = 'none';

            // Start timer
            sessionTimer = setInterval(updateTimer, 1000);

            // Start processing frames at interval
            liveInterval = setInterval(processFrame, DETECTION_INTERVAL_MS);

            // Process first frame immediately
            processFrame();
        } else {
            alert(data.error || 'Failed to start session.');
            stopLiveCamera();
            const startBtn = document.getElementById('btn-start-live');
            startBtn.disabled = false;
            startBtn.innerHTML = '<i class="bi bi-play-fill me-1"></i>Start Detection';
            startBtn.style.display = 'inline-block';
        }
    })
    .catch(function(error) {
        console.error('Session start error:', error);
        alert('Failed to start detection session.');
        stopLiveCamera();
        const startBtn = document.getElementById('btn-start-live');
        startBtn.disabled = false;
        startBtn.innerHTML = '<i class="bi bi-play-fill me-1"></i>Start Detection';
        startBtn.style.display = 'inline-block';
    });
}

/**
 * Process a single frame from the video feed.
 */
function processFrame() {
    if (isProcessing || !liveStream) return;
    isProcessing = true;

    const video = document.getElementById('live-video');
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;

    const ctx = tempCanvas.getContext('2d');
    ctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

    const imageData = tempCanvas.toDataURL('image/jpeg', 0.7);

    fetch('/live/process-frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image: imageData,
            session_id: sessionId,
            db_session_id: dbSessionId
        })
    })
    .then(r => r.json())
    .then(function(data) {
        isProcessing = false;

        if (data.success) {
            if (data.face_detected) {
                // Update emotion display
                document.getElementById('live-emotion').textContent = data.emotion;
                document.getElementById('live-confidence').textContent =
                    data.confidence.toFixed(2) + '%';

                // Draw bounding box
                drawFaceOverlay(data.bbox);

                // Update emotion counts
                emotionCounts[data.emotion] = (emotionCounts[data.emotion] || 0) + 1;
                totalLiveDetections++;

                // Update distribution
                updateEmotionDistribution();
            } else {
                // No face detected — clear overlay
                clearOverlay();
                document.getElementById('live-emotion').textContent = 'No face detected';
                document.getElementById('live-confidence').textContent = '—';
            }
        }
    })
    .catch(function(error) {
        isProcessing = false;
        console.error('Frame processing error:', error);
    });
}

/**
 * Draw a bounding box on the canvas overlay.
 */
function drawFaceOverlay(bbox) {
    const overlay = document.getElementById('live-overlay');
    const video = document.getElementById('live-video');
    const ctx = overlay.getContext('2d');

    // Scale factor between video resolution and display size
    const scaleX = overlay.width / video.videoWidth;
    const scaleY = overlay.height / video.videoHeight;

    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (bbox) {
        const x = bbox.x * scaleX;
        const y = bbox.y * scaleY;
        const w = bbox.w * scaleX;
        const h = bbox.h * scaleY;

        // Draw rectangle
        ctx.strokeStyle = '#27AE60';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        // Draw emotion label background
        const emotion = document.getElementById('live-emotion').textContent;
        ctx.fillStyle = '#27AE60';
        ctx.fillRect(x, y - 25, w, 25);
        ctx.fillStyle = '#FFFFFF';
        ctx.font = 'bold 14px Inter, sans-serif';
        ctx.fillText(emotion, x + 5, y - 7);
    }
}

/**
 * Clear the canvas overlay.
 */
function clearOverlay() {
    const overlay = document.getElementById('live-overlay');
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);
}

/**
 * Update the session timer display.
 */
function updateTimer() {
    if (!sessionStartTime) return;

    const elapsed = Math.floor((new Date() - sessionStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
    const seconds = (elapsed % 60).toString().padStart(2, '0');
    document.getElementById('session-timer').textContent = minutes + ':' + seconds;
}

/**
 * Update the emotion distribution display.
 */
function updateEmotionDistribution() {
    const container = document.getElementById('emotion-distribution');

    if (totalLiveDetections === 0) {
        container.innerHTML = '<div class="empty-state py-3"><p class="text-muted mb-0">No detections yet</p></div>';
        return;
    }

    let html = '';
    // Sort by count descending
    const sorted = Object.entries(emotionCounts).sort((a, b) => b[1] - a[1]);

    for (const [emotion, count] of sorted) {
        const percentage = ((count / totalLiveDetections) * 100).toFixed(1);
        const barColor = percentage > 50 ? 'bg-success' : (percentage > 20 ? 'bg-info' : 'bg-secondary');

        html += `
            <div class="probability-bar">
                <div class="label">
                    <span><span class="badge-emotion badge-${emotion}">${emotion}</span></span>
                    <span>${percentage}%</span>
                </div>
                <div class="progress">
                    <div class="progress-bar ${barColor}" style="width: ${percentage}%"></div>
                </div>
            </div>
        `;
    }

    container.innerHTML = html;
}

/**
 * Stop live emotion detection session.
 */
function stopLiveDetection() {
    // Stop the processing interval
    if (liveInterval) {
        clearInterval(liveInterval);
        liveInterval = null;
    }

    // Stop the timer
    if (sessionTimer) {
        clearInterval(sessionTimer);
        sessionTimer = null;
    }

    // Clear the overlay
    clearOverlay();

    // Hide recording badge
    document.getElementById('live-recording-badge').style.display = 'none';

    // End the session on the server
    if (dbSessionId) {
        fetch('/live/end-session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                db_session_id: dbSessionId
            })
        })
        .then(r => r.json())
        .then(function(data) {
            if (data.success && data.summary) {
                showSessionSummary(data.summary);
            }
        })
        .catch(function(error) {
            console.error('Session end error:', error);
        });
    }

    // Stop camera
    stopLiveCamera();

    // Reset buttons
    const startBtn = document.getElementById('btn-start-live');
    startBtn.style.display = 'inline-block';
    startBtn.disabled = false;
    startBtn.innerHTML = '<i class="bi bi-play-fill me-1"></i>Start Detection';
    document.getElementById('btn-stop-live').style.display = 'none';
}

/**
 * Stop the camera stream.
 */
function stopLiveCamera() {
    if (liveStream) {
        liveStream.getTracks().forEach(function(track) {
            track.stop();
        });
        liveStream = null;
    }
    const video = document.getElementById('live-video');
    if (video) video.srcObject = null;
}

/**
 * Show the session summary card.
 */
function showSessionSummary(summary) {
    const card = document.getElementById('session-summary-card');
    card.style.display = 'block';

    // Format duration
    const dur = summary.duration_seconds || 0;
    const mins = Math.floor(dur / 60).toString().padStart(2, '0');
    const secs = (dur % 60).toString().padStart(2, '0');

    document.getElementById('summary-duration').textContent = mins + ':' + secs;
    document.getElementById('summary-detections').textContent = summary.total_detections || 0;
    document.getElementById('summary-emotion').textContent = summary.dominant_emotion || '—';
    document.getElementById('summary-confidence').textContent =
        summary.average_confidence ? summary.average_confidence.toFixed(2) + '%' : '—';
}
