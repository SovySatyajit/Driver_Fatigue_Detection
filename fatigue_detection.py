# fatigue_detection.py
"""
Driver Monitoring Prototype (Python + MediaPipe + OpenCV)

Features:
- Uses MediaPipe Face Mesh (468 landmarks)
- Computes EAR from eye landmarks
- Head pose estimation (yaw/pitch/roll) via solvePnP
- Simple blink detection and blink-rate smoothing
- Combines signals into a drowsiness score and issues warnings:
    - Yellow (Drowsy) warning
    - Red (Sleeping / Critical) warning

"""
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import pygame

pygame.mixer.init()
SOUND_AVAILABLE = True
ALARM_SOUND_FILE = "alarm.wav"

# ---------------------------
# Parameters (tune these)
# ---------------------------
EAR_THRESHOLD = 0.23                # typical ear threshold for "closed"
EAR_DROWSY_BUFFER = 0.10            # ear < EAR_THRESHOLD + buffer -> drowsy zone
CONSEC_FRAMES_DROWSY = 8            # frames in drowsy zone -> drowsy state
CONSEC_FRAMES_SLEEP = 15            # frames eyes closed -> sleeping state
FPS = 30                            # fallback fps (we read actual from camera if possible)

HEAD_YAW_THRESHOLD = 20.0           # degrees — looking away threshold (yaw)
HEAD_PITCH_THRESHOLD = 15.0         # degrees — nodding down threshold (pitch)

# Combined score thresholds
DROWSY_SCORE_THRESHOLD = 0.4
SLEEP_SCORE_THRESHOLD = 0.7

# ---------------------------
# MediaPipe setup
# ---------------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False,
                             max_num_faces=1,
                             refine_landmarks=True,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Eye landmark indices from MediaPipe face mesh (commonly used)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]     # outer,top,top,inner,bot,bot
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Landmarks for head pose (2D image points). We'll pick:
# nose tip, chin-like, left eye outer, right eye outer, left mouth corner, right mouth corner
POSE_LANDMARKS = {
    "nose_tip": 1,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
    "chin": 199  # approximate lower face point
}

# 3D model coordinates for the pose landmarks (approximate).
# Units are arbitrary but must match the relative positions.
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),        # nose tip
    (0.0, -60.0, -10.0),    # chin (approx)
    (-40.0, 30.0, -30.0),   # left eye outer
    (40.0, 30.0, -30.0),    # right eye outer
    (-30.0, -30.0, -30.0),  # left mouth
    (30.0, -30.0, -30.0)    # right mouth
], dtype=np.float64)

# ---------------------------
# Helpers
# ---------------------------
def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def eye_aspect_ratio(eye_points):
    # eye_points: list of 6 (x,y)
    A = euclidean(eye_points[1], eye_points[5])
    B = euclidean(eye_points[2], eye_points[4])
    C = euclidean(eye_points[0], eye_points[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

ALARM_SOUND_FILE = "alarm.wav"
def play_sound_once():
    if ALARM_SOUND_FILE is None:
        return
    try:
        pygame.mixer.music.load(ALARM_SOUND_FILE)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Error playing sound: {e}")

def get_head_pose(img_pts, size):
    # img_pts: list of 6 2D image points corresponding to MODEL_POINTS order
    # size: (width, height)
    w, h = size
    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4,1))  # assume no lens distortion
    # solvePnP
    success, rotation_vector, translation_vector = cv2.solvePnP(MODEL_POINTS,
                                                                np.array(img_pts, dtype=np.float64),
                                                                camera_matrix, dist_coeffs,
                                                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None
    # Convert rotation vector to Euler angles (in degrees)
    rmat, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rmat[0,0] * rmat[0,0] + rmat[1,0] * rmat[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(rmat[2,1], rmat[2,2])
        y = math.atan2(-rmat[2,0], sy)
        z = math.atan2(rmat[1,0], rmat[0,0])
    else:
        x = math.atan2(-rmat[1,2], rmat[1,1])
        y = math.atan2(-rmat[2,0], sy)
        z = 0
    # convert to degrees
    pitch = math.degrees(x)
    yaw = math.degrees(y)
    roll = math.degrees(z)
    return (yaw, pitch, roll)

# ---------------------------
# Main loop
# ---------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # read FPS if available
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = actual_fps if actual_fps and actual_fps > 1 else FPS
    frame_time = 1.0 / fps

    # State counters and trackers
    sleep_counter = 0
    drowsy_counter = 0
    active_counter = 0
    blink_timestamps = []
    last_blink = 0
    alarm_triggered = False
    last_alert_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)
        status_text = "No face"
        color = (200, 200, 200)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            # Convert landmarks to (x,y) pixel coords
            lm = []
            for p in face_landmarks.landmark:
                lm.append((int(p.x * w), int(p.y * h)))

            # Eye landmarks
            left_eye_pts = [lm[i] for i in LEFT_EYE_IDX]
            right_eye_pts = [lm[i] for i in RIGHT_EYE_IDX]

            # Draw small points
            for (x,y) in left_eye_pts + right_eye_pts:
                cv2.circle(frame, (x,y), 1, (255,0,0), -1)

            # EAR
            left_ear = eye_aspect_ratio(left_eye_pts)
            right_ear = eye_aspect_ratio(right_eye_pts)
            ear = (left_ear + right_ear) / 2.0

            # Head pose: build image points in correct order for solvePnP:
            img_pts = [
                lm[POSE_LANDMARKS["nose_tip"]],
                lm[POSE_LANDMARKS["chin"]],
                lm[POSE_LANDMARKS["left_eye_outer"]],
                lm[POSE_LANDMARKS["right_eye_outer"]],
                lm[POSE_LANDMARKS["left_mouth"]],
                lm[POSE_LANDMARKS["right_mouth"]]
            ]

            head_pose = get_head_pose(img_pts, (w, h))
            if head_pose is None:
                yaw = pitch = roll = 0.0
            else:
                yaw, pitch, roll = head_pose  # yaw left/right, pitch up/down, roll tilt

            # Blink detection (very simple): detect quick closure
            # When ear goes below small value briefly -> record blink
            BLINK_EAR = 0.18
            now = time.time()
            if ear < BLINK_EAR:
                # debounce quick repeated frames for single blink
                if now - last_blink > 0.15:
                    blink_timestamps.append(now)
                    last_blink = now
                    # keep only last 20 blinks
                    if len(blink_timestamps) > 20:
                        blink_timestamps = blink_timestamps[-20:]

            # compute blink rate (blinks per minute) using last N seconds
            blink_rate = 0.0
            if len(blink_timestamps) >= 2:
                window = now - blink_timestamps[0]
                if window > 0:
                    blink_rate = len(blink_timestamps) * 60.0 / window

            # Determine eye state using EAR thresholds
            if ear < EAR_THRESHOLD:
                sleep_counter += 1
                drowsy_counter = 0
                active_counter = 0
            elif ear < EAR_THRESHOLD + EAR_DROWSY_BUFFER:
                drowsy_counter += 1
                sleep_counter = 0
                active_counter = 0
            else:
                active_counter += 1
                sleep_counter = 0
                drowsy_counter = 0

            # Convert counters into times (seconds)
            sleep_time = sleep_counter * frame_time
            drowsy_time = drowsy_counter * frame_time
            active_time = active_counter * frame_time

            # Compose a drowsiness score [0..1]
            # Components:
            # - eye_closure_score: 0 (open) .. 1 (closed)
            eye_closure_score = np.clip((EAR_THRESHOLD - ear) / (EAR_THRESHOLD), 0.0, 1.0)
            # - head_pose_score: how much yaw/pitch exceed thresholds
            head_yaw_score = max(0.0, (abs(yaw) - HEAD_YAW_THRESHOLD) / 30.0)  # normalized
            head_pitch_score = max(0.0, (abs(pitch) - HEAD_PITCH_THRESHOLD) / 30.0)
            head_pose_score = np.clip(max(head_yaw_score, head_pitch_score), 0.0, 1.0)
            # - blinkiness: unusually low blink rate or long closures can indicate microsleep
            # Here we penalize very low blink rate (e.g., <6 bpm) as possible fatigue,
            # and very high blink_rate could also be sign of tired eyes — we keep simple.
            blink_score = 0.0
            if blink_rate < 8:
                blink_score = np.clip((8.0 - blink_rate) / 8.0, 0.0, 1.0)

            # Weighted sum
            drowsy_score = 0.6 * eye_closure_score + 0.3 * head_pose_score + 0.1 * blink_score
            drowsy_score = float(np.clip(drowsy_score, 0.0, 1.0))

            # Decide state based on counters and combined score
            status = "Active"
            color = (0, 255, 0)

            # Sleep (red) if eyes closed for enough consecutive frames OR very high score sustained
            if sleep_counter >= CONSEC_FRAMES_SLEEP or (drowsy_score > SLEEP_SCORE_THRESHOLD and sleep_time > 0.3):
                status = "SLEEPING !!!"
                color = (0, 0, 255)
                # trigger alarm (throttled)
                if not alarm_triggered or (time.time() - last_alert_time) > 5.0:
                    if SOUND_AVAILABLE and ALARM_SOUND_FILE:
                        play_sound_once()
                    alarm_triggered = True
                    last_alert_time = time.time()
            # Drowsy (yellow) if drowsy counter reached OR drowsy_score exceeded
            elif drowsy_counter >= CONSEC_FRAMES_DROWSY or drowsy_score > DROWSY_SCORE_THRESHOLD:
                status = "Drowsy!"
                color = (0, 255, 255)
                alarm_triggered = False
            else:
                status = "Active :)"
                color = (0, 255, 0)
                alarm_triggered = False

            # Draw info on frame
            cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"EAR: {ear:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
            cv2.putText(frame, f"Dscore: {drowsy_score:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
            cv2.putText(frame, f"Yaw:{yaw:.1f} Pitch:{pitch:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
            cv2.putText(frame, f"Blink/min: {blink_rate:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

            # Visualize a rectangle or circle showing status
            cv2.rectangle(frame, (5, 5), (220, 170), color, 2)

            # Optionally draw the face mesh (uncomment to visualize)
            # mp_drawing.draw_landmarks(frame, face_landmarks, mp_face.FACEMESH_CONTOURS,
            #                           mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
            #                           mp_drawing.DrawingSpec(color=(0,0,255), thickness=1))
        else:
            cv2.putText(frame, "No face detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow("Driver Monitoring Prototype", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
