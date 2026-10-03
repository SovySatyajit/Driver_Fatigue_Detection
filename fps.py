import cv2

cap = cv2.VideoCapture(0)  # Open default camera

if not cap.isOpened():
    print("Cannot open camera")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera FPS: {fps}")

cap.release()
