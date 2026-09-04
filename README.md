# Emotion Detection Using Facial Expression

A web application that detects human emotions from facial expressions using computer vision and deep learning. Built as an academic project with Python, Flask, OpenCV, and TensorFlow/Keras.

---

## 📋 Features

- **User Authentication** — Register, login, logout with secure password hashing
- **Image Emotion Detection** — Upload an image or capture from camera
- **Live Camera Detection** — Real-time emotion detection via webcam
- **Face Detection** — OpenCV Haar Cascade for reliable face detection
- **Emotion Classification** — CNN model classifies 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Result History** — View and manage past detection results
- **Live Session Tracking** — Session summaries with dominant emotion and statistics
- **Excel Export** — Download detection history as a formatted `.xlsx` report
- **User Isolation** — Each user can only access their own data

---

## 🏗 Architecture

```
Browser → Flask Routes → Application Services → SQLAlchemy → SQLite Database
                              ├── Authentication
                              ├── Face Detection (OpenCV)
                              ├── Emotion Prediction (Keras CNN)
                              ├── Result Management
                              ├── Live Detection
                              └── Excel Export (openpyxl)
```

**ML Pipeline:**
```
Input Image/Camera → Face Detection → Face Crop → Preprocessing → Emotion Model → Emotion + Confidence → Save → Display
```

---

## 🛠 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Backend | Python 3, Flask | Web framework |
| Database | SQLite, SQLAlchemy | Data persistence |
| Authentication | Flask-Login, Werkzeug | Session management, password hashing |
| Face Detection | OpenCV | Haar Cascade face detection |
| Emotion Model | TensorFlow/Keras | CNN emotion classification |
| Frontend | HTML5, CSS3, JavaScript, Bootstrap 5 | User interface |
| Export | openpyxl | Excel report generation |

---

## 📁 Folder Structure

```
emotion_detection_project/
├── app.py                    # Main Flask application
├── config.py                 # Centralized configuration
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── model/                    # ML model directory
│   └── emotion_model.h5      # ⬅ Place your model here
│
├── database/                 # SQLite database (auto-created)
│   └── emotion_app.db
│
├── models/                   # SQLAlchemy ORM models
│   ├── user.py               # User model
│   ├── detection.py          # Detection model
│   └── live_session.py       # LiveSession model
│
├── routes/                   # Flask Blueprints
│   ├── auth.py               # Login/Register/Logout
│   ├── dashboard.py          # Home, Profile, Conclusion
│   ├── detection.py          # Image/Camera/Live detection
│   └── export.py             # Excel export
│
├── utils/                    # Utility modules
│   ├── database.py           # Database initialization
│   ├── face_detector.py      # OpenCV face detection
│   ├── emotion_predictor.py  # Keras model loading + prediction
│   └── excel_exporter.py     # Excel report generator
│
├── templates/                # Jinja2 HTML templates
│   ├── base.html             # Base layout
│   ├── login.html            # Login page
│   ├── register.html         # Registration page
│   ├── home.html             # Dashboard
│   ├── user_module.html      # Profile + History
│   ├── face_detection.html   # Upload/Capture detection
│   ├── result.html           # Detection result
│   ├── live_detection.html   # Live webcam detection
│   ├── conclusion.html       # Academic conclusion
│   └── error.html            # Error pages
│
└── static/
    ├── css/style.css         # Custom styles
    ├── js/
    │   ├── main.js           # Common utilities
    │   ├── camera.js         # Camera capture
    │   └── live_detection.js # Live detection
    ├── uploads/              # Uploaded images (auto-created)
    └── results/              # Processed images (auto-created)
```

---

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.10 or 3.11** (required for TensorFlow compatibility)
- **pip** (Python package manager)
- **Webcam** (optional, for camera and live detection features)

### Step 1: Clone or Download the Project

Place the project files in your desired directory.

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** TensorFlow is a large package (~500 MB+). Installation may take several minutes.

### Step 5: Place the Emotion Model

The application requires a pre-trained Keras model file. **No fallback or demo mode exists** — the model is mandatory for emotion detection features.

**Download a FER-2013 model from one of these sources:**

**Option A — Hugging Face:**
```python
# Run this in a Python script or terminal
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="shivamprasad1001/Emo0.1", filename="Emo0.1.h5")
print(f"Model downloaded to: {model_path}")
# Copy the downloaded file to: model/emotion_model.h5
```

**Option B — GitHub:**
1. Visit: https://github.com/GSNCodes/Emotion-Detection-FER2013
2. Download the pre-trained `.h5` model file
3. Place it at: `model/emotion_model.h5`

**Model Requirements:**
- Format: `.h5` or `.keras`
- Input shape: `(48, 48, 1)` — 48×48 grayscale
- Output: 7 classes in order: `[Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral]`
- If your model uses a different configuration, update `config.py`

### Step 6: Run the Application

```bash
python app.py
```

The application will start at: **http://127.0.0.1:5000**

---

## 🔧 Configuration

All settings are centralized in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_PATH` | `model/emotion_model.h5` | Path to the Keras model |
| `MODEL_INPUT_SIZE` | `(48, 48)` | Expected input dimensions |
| `MODEL_COLOR_MODE` | `grayscale` | Input color mode |
| `EMOTION_LABELS` | 7 FER-2013 classes | Emotion class labels and order |
| `LIVE_DETECTION_INTERVAL` | `2` seconds | Frame processing interval |
| `MAX_CONTENT_LENGTH` | `16 MB` | Maximum upload file size |

---

## 📸 Camera Permissions

- The browser will request camera access when you use "Capture from Camera" or "Live Detection"
- Click **Allow** when prompted
- For HTTPS requirements: `localhost` / `127.0.0.1` is automatically allowed
- If denied, check your browser's site settings to re-enable camera access

---

## 📊 Excel Export

- Go to **Profile** → **Export Results** or **Dashboard** → **Export Excel**
- Downloads an `.xlsx` file with two worksheets:
  1. **Detection History** — All saved image/camera detections
  2. **Live Sessions** — All live detection session summaries
- Only your data is exported (user isolation enforced)
- Professional formatting: bold headers, filters, frozen panes, borders

---

## 🗃 Database

- **Engine:** SQLite (file-based, no setup required)
- **ORM:** SQLAlchemy
- **Location:** `database/emotion_app.db` (auto-created on first run)
- **Tables:** `users`, `detections`, `live_sessions`

To reset the database, simply delete `database/emotion_app.db` and restart the app.

---

## ❓ Troubleshooting

### "Emotion model unavailable"
- Ensure the model file exists at `model/emotion_model.h5`
- Check the console output for specific validation errors
- Verify the model's input/output shapes match `config.py`

### TensorFlow installation fails
- Ensure Python 3.10 or 3.11 (not 3.12+)
- Try: `pip install tensorflow --no-cache-dir`
- On Windows, ensure Visual C++ Redistributable is installed

### Camera not working
- Allow camera access in browser settings
- Try Chrome or Edge (best WebRTC support)
- Check if another application is using the camera

### "No face detected"
- Ensure the face is clearly visible and well-lit
- Try a frontal face photo (Haar Cascade works best with frontal faces)
- Avoid extreme angles or heavy occlusion

---

## 🎓 Academic Project Explanation

This project demonstrates the practical application of:

1. **Computer Vision** — Using OpenCV Haar Cascades for real-time face detection
2. **Deep Learning** — Convolutional Neural Network for emotion classification
3. **Web Development** — Full-stack Flask application with authentication
4. **Database Management** — SQLite with SQLAlchemy ORM
5. **Data Export** — Generating formatted Excel reports with openpyxl

**Key concepts for viva:**
- Haar Cascade: A machine learning-based approach for object detection using edge and rectangle features
- CNN: Learns hierarchical features from images (edges → textures → facial patterns → emotions)
- FER-2013: A benchmark dataset of 35,000+ labeled facial expression images
- The system classifies *visible facial expressions*, not true internal emotions

---

## ⚠ Disclaimer

This is an academic demonstration of facial expression classification. Detected "emotions" are statistical predictions based on visible facial patterns and may not accurately reflect a person's true internal emotional state.

---

## 📄 License

Academic project — for educational purposes only.
