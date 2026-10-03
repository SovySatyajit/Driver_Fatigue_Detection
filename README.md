<h1 align="center">😴 Driver Fatigue Detection</h1>

<h3 align="center">Real-time AI-based driver monitoring using facial landmarks and eye movement analysis</h3>

<p align="center">
  <img src="https://img.shields.io/badge/PYTHON-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OPENCV-COMPUTER%20VISION-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/MEDIAPIPE-FACE%20MESH-00BCD4?style=for-the-badge&logo=google&logoColor=white" alt="MediaPipe">
  <br>
  <img src="https://img.shields.io/badge/NUMPY-NUMERICAL%20COMPUTING-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/PYGAME-AUDIO%20ALERTS-00AEEF?style=for-the-badge" alt="Pygame">
  <img src="https://img.shields.io/badge/LICENSE-MIT-green?style=for-the-badge" alt="License">
</p>

---

## 📖 Introduction

**Driver Fatigue Detection** is a real-time computer vision system that monitors a driver's alertness by analyzing facial landmarks and eye movement through a webcam feed. Using **MediaPipe's Face Mesh** for landmark detection and the **Eye Aspect Ratio (EAR)** algorithm, the system continuously classifies the driver's state into one of three categories — **Active**, **Drowsy**, or **Sleeping** — and triggers an audio alarm if signs of sleep persist beyond a safe threshold.

This project was built independently, end-to-end, covering face/landmark detection, signal-based state classification, real-time video processing, and alert handling.

---

## 🎯 Problem Statement

Driver fatigue is one of the leading causes of road accidents worldwide. Unlike alcohol impairment or speeding, drowsiness is difficult to self-detect — a driver is often unaware they are falling asleep until it is too late. Most vehicles, especially in the public and commercial transport sector, have **no built-in mechanism to detect or warn against driver drowsiness**.

**Driver Fatigue Detection** addresses this gap with a lightweight, camera-based monitoring system that:

- Requires no specialized hardware — works with any standard webcam
- Runs entirely on-device in real time
- Provides an audible alarm before the driver reaches a dangerous state of sleep

---

## ⚙️ Working

The system processes every webcam frame through the same five-stage pipeline:

```
 Webcam  ──►  Face Mesh  ──►  Measure  ──►  Decide  ──►  Output
 (frames)    (landmarks)    (EAR, head    (score +     (status on
                             pose, blinks)  rules)      screen + alarm)
```

### 1️⃣ Face and landmark detection
Each frame is converted from BGR to RGB and passed to **MediaPipe Face Mesh**, which returns facial landmarks (468 points, 478 with iris refinement). Only the points needed for the eyes, nose, chin and mouth are used.

### 2️⃣ Eye Aspect Ratio (EAR)
Six landmarks around each eye give two vertical distances (A, B) and one horizontal distance (C). EAR is the ratio between them:

```
EAR = (A + B) / (2 × C)
```

An **open eye** gives a high EAR; as the eye **closes**, EAR drops toward zero. The EAR of both eyes is averaged.

### 3️⃣ Blink and head-pose analysis
- **Blinks** — short EAR dips are counted (with a 0.15 s debounce) to estimate blinks per minute.
- **Head pose** — `cv2.solvePnP` estimates yaw, pitch and roll from six facial landmarks, detecting looking away or a nodding head.

### 4️⃣ Drowsiness score
The three signals are combined into one score between **0** (alert) and **1** (very drowsy):

```
D = 0.6 × EyeClosure + 0.3 × HeadPose + 0.1 × Blink
```

### 5️⃣ State classification and alert
Counters over **consecutive frames** prevent a normal blink from raising a false alarm.

| State | Trigger | Output |
|-------|---------|--------|
| 🟢 **Active** | None of the conditions below | Green status |
| 🟡 **Drowsy** | EAR between 0.23 and 0.33 for **8** consecutive frames, **or** score > 0.4 | Yellow status |
| 🔴 **Sleeping** | EAR < 0.23 for **15** consecutive frames, **or** score > 0.7 with eyes closed > 0.3 s | Red status + **audio alarm** |

The alarm is throttled to at most once every 5 seconds so it does not restart on every frame.

---

## ✨ Features

- 👁️ **EAR-based eye closure detection** using MediaPipe Face Mesh
- 😴 **Three-level state classification** — Active, Drowsy, Sleeping
- 🧭 **Head pose estimation** (yaw, pitch, roll) via `solvePnP`
- 😉 **Blink-rate tracking** with debounce
- 📊 **Weighted drowsiness score** from 0 to 1
- 🔔 **Audio alarm** with throttling (Pygame)
- 🖥️ **Live on-screen HUD** showing status, EAR, score, angles and blink rate
- 🔒 **Fully local** — no cloud, no database, no data leaves your machine

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python** | Core implementation language |
| **OpenCV** | Webcam capture, drawing, display, head-pose math |
| **MediaPipe Face Mesh** | Real-time facial landmark detection |
| **NumPy** | Numerical operations and value clipping |
| **Pygame** | Audio alarm playback |

---

## 📁 Project Structure

```
driver-fatigue-detection/
├── fatigue_detection.py   # Main detection pipeline and live display
├── fps.py                 # Camera availability and FPS check
├── alarm.wav              # Alarm sound for the Sleeping state
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── docs/
    └── Driver_Fatigue_Detection_Technical_Documentation.pdf
```

---

## 🚀 Installation

### Prerequisites
- Python **3.9 – 3.12**
- A working webcam
- Speakers or headphones
- `alarm.wav` in the same folder as `fatigue_detection.py`

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/driver-fatigue-detection.git
cd driver-fatigue-detection
```

**2. (Optional) Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python fatigue_detection.py
```

- Sit in front of the webcam with your face clearly visible and well lit.
- Watch the status on screen: 🟢 Active → 🟡 Drowsy → 🔴 Sleeping.
- Press **`ESC`** to quit.

To test that your camera works, run:

```bash
python fps.py
```

### On-screen display

| Item | Meaning |
|------|---------|
| **Status** | Current driver state, colour-coded |
| **EAR** | Current eye openness |
| **Dscore** | Combined drowsiness score (0–1) |
| **Yaw / Pitch** | Head direction in degrees |
| **Blink/min** | Estimated blink rate |

---

## 🎛️ Configuration

All settings are constants at the top of `fatigue_detection.py`.

| Constant | Default | Description |
|----------|---------|-------------|
| `EAR_THRESHOLD` | `0.23` | EAR below this means the eye is closed |
| `EAR_DROWSY_BUFFER` | `0.10` | Width of the drowsy zone above the threshold |
| `CONSEC_FRAMES_DROWSY` | `8` | Frames in the drowsy zone to trigger Drowsy |
| `CONSEC_FRAMES_SLEEP` | `15` | Frames with closed eyes to trigger Sleeping |
| `HEAD_YAW_THRESHOLD` | `20.0` | Left/right turn limit (degrees) |
| `HEAD_PITCH_THRESHOLD` | `15.0` | Up/down nod limit (degrees) |
| `DROWSY_SCORE_THRESHOLD` | `0.4` | Score needed for Drowsy |
| `SLEEP_SCORE_THRESHOLD` | `0.7` | Score needed for Sleeping |
| `ALARM_SOUND_FILE` | `alarm.wav` | Alarm audio file |

> 💡 These values are rules of thumb. Calibrate them for your camera, lighting and face.

---

## 🧰 Troubleshooting

| Problem | Solution |
|---------|----------|
| `Cannot open camera` | Close other apps using the webcam, or try `cv2.VideoCapture(1)` |
| `pygame.error` on start | No audio device found — connect speakers or headphones |
| `module 'mediapipe' has no attribute 'solutions'` | Use Python 3.9–3.12 and `pip install "mediapipe>=0.10,<0.11"` |
| Always shows Drowsy | Improve lighting, face the camera, re-tune thresholds |
| No alarm sound | Check that `alarm.wav` exists next to the script |

---

## ⚠️ Limitations

- Thresholds are hand-tuned heuristics, not learned from data
- Sunglasses, low light, occlusion and extreme head angles reduce accuracy
- Only one face is tracked
- The 3D head model used for pose estimation is approximate
- No data logging, analytics or vehicle integration

---

## 🔮 Future Improvements

- [ ] Move thresholds and weights to a JSON/YAML config
- [ ] Per-driver calibration (baseline EAR and head position)
- [ ] PERCLOS, blink-duration and yawning detection
- [ ] Modular codebase with unit tests
- [ ] Evaluation on labelled data (precision, recall, F1, false alarms)
- [ ] Temporal deep-learning classifier (LSTM / TCN)
- [ ] Session analytics and hardware / vehicle alerts

---

## 🛡️ Safety Disclaimer

This project is a **research and learning prototype**. It is **not a certified automotive safety system** and must not be relied upon while driving a real vehicle.

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**. Add a `LICENSE` file to the repository root.

---

<p align="center">Built with ❤️ using Python, OpenCV and MediaPipe</p>
