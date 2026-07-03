<div align="center">

# 😴 Driver Fatigue Detection

### Real-time AI-based driver monitoring using facial landmarks and eye movement analysis

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-00C4CC?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pygame](https://img.shields.io/badge/Pygame-Audio%20Alerts-00B0FF?style=for-the-badge&logo=python&logoColor=white)](https://www.pygame.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](#-license)

</div>

---

## 📖 Introduction

**Driver Fatigue Detection** is a real-time computer vision system that monitors a driver's alertness by analyzing facial landmarks and eye movement through a webcam feed. Using **MediaPipe's Face Mesh** for landmark detection and the **Eye Aspect Ratio (EAR)** algorithm, the system continuously classifies the driver's state into one of three categories — **Active**, **Drowsy**, or **Sleepy** — and triggers an audio alarm if signs of sleep persist beyond a safe threshold.

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

1. The webcam continuously captures video frames of the driver's face.
2. Each frame is passed through **MediaPipe Face Mesh** to detect 468 facial landmarks.
3. Landmarks corresponding to the eyes are extracted, and the **Eye Aspect Ratio (EAR)** is calculated for both eyes.
4. The EAR value is tracked across frames:
   - **High/stable EAR** → eyes open → classified as **Active**
   - **Reduced EAR held over a short duration** → partial eye closure → classified as **Drowsy**
   - **Very low EAR sustained continuously** → eyes closed → classified as **Sleepy**
5. If the **Sleepy** state persists for **7 or more consecutive seconds**, an **audio alarm** is triggered using Pygame to alert the driver.
6. The driver's current state and EAR value are displayed live on the video feed for continuous visual feedback.

**Eye Aspect Ratio (EAR) formula:**

```
EAR = ( ||p2 - p6|| + ||p3 - p5|| ) / ( 2 × ||p1 - p4|| )
```

Where `p1...p6` are the six eye landmark coordinates. EAR remains fairly constant when the eye is open and drops sharply toward zero as the eye closes — making it a reliable, lightweight signal for drowsiness detection without needing a trained classification model.

---

## 🧠 AI Pipeline

```
┌───────────────────────┐
│   Webcam Video Feed     │
└───────────┬─────────────┘
            ▼
┌───────────────────────┐
│  Frame Preprocessing     │  (resize, color conversion via OpenCV)
└───────────┬─────────────┘
            ▼
┌───────────────────────┐
│  MediaPipe Face Mesh     │  → 468 facial landmarks
└───────────┬─────────────┘
            ▼
┌───────────────────────┐
│  Eye Landmark Extraction │  → Left & right eye coordinates
└───────────┬─────────────┘
            ▼
┌───────────────────────┐
│  EAR Calculation (NumPy) │  → Per-frame eye aspect ratio
└───────────┬─────────────┘
            ▼
┌───────────────────────┐
│  State Classification    │  → Active / Drowsy / Sleepy
│  (Threshold + Duration)  │
└───────────┬─────────────┘
            ▼
┌───────────────────────┐
│  Alarm Trigger (Pygame)  │  → Fires if "Sleepy" ≥ 7 seconds
└───────────────────────┘
```

---

## 🛠️ Technologies

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **OpenCV** | Video capture, frame processing, and on-screen overlays |
| **MediaPipe** | Real-time facial landmark detection (Face Mesh) |
| **NumPy** | Vector math for EAR calculation |
| **Pygame** | Audio playback for the alarm system |

---

## 📂 Folder Structure

```
driver-fatigue-detection/
├── main.py                  # Entry point — runs the real-time detection loop
├── utils/
│   ├── ear_calculator.py    # Eye Aspect Ratio computation logic
│   ├── landmarks.py         # Eye landmark index mapping for MediaPipe Face Mesh
│   └── alarm.py             # Pygame-based audio alarm handler
├── assets/
│   ├── alarm.mp3             # Alarm sound file
│   └── demo.gif               # Demo preview (see below)
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- A working webcam

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/driver-fatigue-detection.git
cd driver-fatigue-detection

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python main.py
```

**`requirements.txt`**
```
opencv-python
mediapipe
numpy
pygame
```

Press **`q`** at any time to exit the application.

---

## 🎬 Demo

> _Add a short GIF here showing the system detecting Active, Drowsy, and Sleepy states with the alarm triggering._

```
assets/demo.gif
```

```markdown
![Driver Fatigue Detection Demo](assets/demo.gif)
```

---

## 🔮 Future Scope

- 📱 Deploy as a mobile app using TensorFlow Lite for in-vehicle use
- 🌙 Improve low-light and infrared performance for night driving
- 🎯 Add yawning detection as an additional fatigue signal
- 📊 Log driver state history for fleet safety analytics
- 🔔 Integrate with vehicle systems (e.g., seat vibration, dashboard alerts)
- ☁️ Add cloud sync for fleet-wide monitoring dashboards
- 🧠 Explore a trained CNN-based classifier as an alternative to threshold-based EAR for higher robustness across diverse faces

---

## 📚 Learnings

Building this project independently helped strengthen:

- Real-time computer vision pipelines using OpenCV
- Facial landmark detection with MediaPipe Face Mesh
- Signal-based state classification without a trained ML model
- Threshold tuning and temporal logic (duration-based state changes)
- Audio event handling in Python with Pygame
- Structuring a computer vision project for readability and reuse

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**[Your Name]**
AI Engineer — Driver Fatigue Detection (Independent Project)

- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)
- Email: your.email@example.com

<div align="center">

⭐ If you found this project useful, consider giving it a star!

</div>
