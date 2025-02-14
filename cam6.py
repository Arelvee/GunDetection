import cv2
import torch
import threading
import os
import pygame  # For sound alarm
from ultralytics import YOLO

# Check if AMD ROCm (or CUDA) is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

current_dir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(current_dir, "best.pt")
sound_path = os.path.join(current_dir, "notif.mp3")  # Path to alarm sound

# Load the YOLO model globally
model = YOLO(model_path).to(device)  # Ensure model is accessible to all functions

# Store last detected bounding boxes
last_detected_boxes = []
frame_persistence = 10  # Frames to keep boxes visible
frames_since_last_detection = 0  # Track frames without detection
alarm_triggered = False  # Flag to avoid repeated alarms

# List of firearm-related class labels that should trigger the alarm
firearm_classes = {"unregistered firearm"}  # Adjust based on your YOLO model's class names

# Initialize pygame mixer for sound notifications
pygame.mixer.init()

def play_alarm():
    """Plays the notification sound if not already playing."""
    global alarm_triggered
    if not alarm_triggered:
        sound = pygame.mixer.Sound(sound_path)
        sound.play()
        alarm_triggered = True  # Set flag to prevent repeated alarms

def perform_detection(frame, confidence_threshold=0.6):
    """Runs YOLO detection and returns the processed frame with bounding boxes."""
    global last_detected_boxes, frame_persistence, frames_since_last_detection, alarm_triggered

    # Run YOLO detection
    results = model(frame)
    detections = results[0].boxes

    new_detected_boxes = []
    firearm_detected = False  # Track if any firearm is detected

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
        conf = detection.conf[0].item()
        class_id = int(detection.cls[0].item())  # Get class ID
        label = model.names[class_id]  # Convert class ID to label

        if conf >= confidence_threshold:
            new_detected_boxes.append((x1, y1, x2, y2, conf, label))

            # Check if detected object is an unregistered firearm
            if label.lower() in firearm_classes:
                firearm_detected = True

    # Trigger alarm if a firearm is detected
    if firearm_detected and not alarm_triggered:
        threading.Thread(target=play_alarm, daemon=True).start()
        alarm_triggered = True
    elif not firearm_detected:
        alarm_triggered = False  # Reset alarm flag when no firearm is detected

    # Update last detected boxes
    if new_detected_boxes:
        last_detected_boxes = new_detected_boxes
        frame_persistence = 10  # Reset persistence
        frames_since_last_detection = 0  # Reset missing frame counter
    else:
        frames_since_last_detection += 1

    # If no detections for too long, clear bounding boxes
    if frames_since_last_detection > frame_persistence:
        last_detected_boxes = []

    # Draw bounding boxes
    for (x1, y1, x2, y2, conf, label) in last_detected_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green box
        text = f"{label} | mAP: {conf:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def display_frames(vid_path, confidence_threshold=0.6):
    cap = cv2.VideoCapture(vid_path)

    if not cap.isOpened():
        print("[ERROR] Could not open video source!")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("[ERROR] Failed to read frame")
            break

        frame = perform_detection(frame, confidence_threshold)

        # Display the frame
        cv2.imshow("YOLO Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main(vid_path=0):
    print("Starting video processing...")
    display_frames(vid_path, confidence_threshold=0.6)  # Run directly without threading

if __name__ == "__main__":
    main()