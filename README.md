# Driver Fatigue Detection

A real-time driver drowsiness detector. It watches your face through a webcam, measures how open your eyes are, how often you blink and where your head is pointing, and warns you when you look drowsy or fall asleep.

> Research prototype. Not a certified safety system. Do not rely on it while driving.

---

## What it does

- Finds your face with **MediaPipe Face Mesh**
- Measures eye openness with **EAR (Eye Aspect Ratio)**
- Counts blinks and estimates blinks per minute
- Estimates head direction (yaw, pitch, roll) with OpenCV `solvePnP`
- Blends everything into a **drowsiness score** from 0 to 1
- Shows one of three states on the live video:
  - Active (green)
  - Drowsy (yellow)
  - Sleeping (red, plays an alarm sound)

## How it works

```
Webcam -> Face landmarks -> Measure (EAR, head pose, blinks) -> Score + rules -> Display + alarm
```

Drowsiness score:

```
D = 0.6 * eye_closure + 0.3 * head_pose + 0.1 * blink_score
```

| State    | When it triggers                                                                 |
|----------|----------------------------------------------------------------------------------|
| Sleeping | Eyes closed (EAR < 0.23) for 15 frames in a row, or score > 0.7 with eyes closed > 0.3 s |
| Drowsy   | EAR between 0.23 and 0.33 for 8 frames in a row, or score > 0.4                  |
| Active   | Anything else                                                                    |

## Requirements

- Python 3.9 - 3.12
- A webcam
- Speakers or headphones
- `alarm.wav` in the same folder as the script

## Setup and run (VS Code terminal)

```bash

# 1. Setup
pip install opencv-python mediapipe numpy pygame
# 2. RUN
python fatigue_detection.py


Press **ESC** or **Ctrl+C** in the terminal window to quit.

Tip: in VS Code if option asked, choose **Python: Interpreter**, 

To check your camera first:
python fps.py

```

## Settings you can change

Open `fatigue_detection.py` and edit the constants near the top.

| Constant                 | Default | Meaning                                  |
|--------------------------|---------|------------------------------------------|
| `EAR_THRESHOLD`          | 0.23    | Below this, the eye counts as closed     |
| `EAR_DROWSY_BUFFER`      | 0.10    | Width of the "drowsy" EAR zone           |
| `CONSEC_FRAMES_DROWSY`   | 8       | Frames in a row needed for Drowsy        |
| `CONSEC_FRAMES_SLEEP`    | 15      | Frames in a row needed for Sleeping      |
| `HEAD_YAW_THRESHOLD`     | 20      | Degrees of left/right turn allowed       |
| `HEAD_PITCH_THRESHOLD`   | 15      | Degrees of up/down nod allowed           |
| `DROWSY_SCORE_THRESHOLD` | 0.4     | Score needed for Drowsy                  |
| `SLEEP_SCORE_THRESHOLD`  | 0.7     | Score needed for Sleeping                |
| `ALARM_SOUND_FILE`       | alarm.wav | Alarm sound                            |

These values are rules of thumb. Adjust them for your camera, lighting and face.

## Files

```
fatigue_detection.py   Main program (detection + live display)
fps.py                 Camera and FPS check
alarm.wav              Alarm sound
requirements.txt       Python packages
README.md              This file
```

## Troubleshooting

| Problem                          | What to try                                                       |
|----------------------------------|-------------------------------------------------------------------|
| "Cannot open camera"             | Close other apps using the webcam, or try `cv2.VideoCapture(1)`   |
| `pygame.error` at start          | No audio device found. Connect speakers or headphones             |

| `mediapipe has no attribute      
'solutions'                        | Use Python 3.9-3.12 and `pip install "mediapipe>=0.10,<0.11"`

| Always shows Drowsy              | Improve lighting, face the camera, re-tune thresholds             |
| No sound                         | Check that `alarm.wav` is next to the script                      |

## Limitations

- Thresholds are hand-tuned, not learned from data
- Sunglasses, dark rooms and extreme head angles reduce accuracy
- Only one face is tracked
- No data is stored