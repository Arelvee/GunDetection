
import sqlite3
from datetime import datetime
from ultralytics import YOLO
import cv2

import torch

import threading
import tkinter as tk
from tkinter import messagebox
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

notification_displayed = False  # Global flag to track if a notification is being displayed

def show_notification(message):
    global notification_displayed
    if not notification_displayed:
        notification_displayed = True
        # Create a new Tk instance
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        # Display the notification message box
        messagebox.showinfo("CCTV3", message)
        # Reset the flag after the message box is closed
        notification_displayed = False
        # Run the Tkinter main loop
        root.mainloop()


def perform_detection(model, frame,confidence_threshold):

    # Initialize SQLite connection and cursor
    conn = sqlite3.connect('mydb.db')
    cursor = conn.cursor()
    # Detect and track objects
    results = model.track(frame, persist=True, imgsz=640, device='cuda')
    filtered_results = [detection for detection in results[0].boxes if detection.conf > confidence_threshold]

    # Create a copy of the original color frame
    frame_with_boxes = results[0].plot(boxes=filtered_results)

    if filtered_results:
        frame = results[0].plot(boxes=filtered_results)
        # out.write(frame)
        # Save detected frame to SQLite database
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        image_data = cv2.imencode('.jpg', frame)[1].tobytes()
        cursor.execute('INSERT INTO detected_frames (timestamp, image_data) VALUES (?, ?)',
                       (timestamp, image_data))
        conn.commit()
        # Display frame (optional)
        print("Object detected with confidence >= {}! Frame saved.".format(confidence_threshold))
        threading.Thread(target=show_notification, args=("Gun Detected!",)).start()
    return results,frame_with_boxes
def display_frames(vid_path, model, confidence_threshold,image_name="captured_image.jpg"):
    cap = cv2.VideoCapture(vid_path)
    cv2.namedWindow("cam3", cv2.WINDOW_NORMAL)
    frame_no = 1
    skip_frame = 5  # Skip every 5 frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Failed to capture frame from camera")
            break

        # Perform YOLOv5 detection only for every 5th frame
        if frame_no % skip_frame == 0:
            results, frame_with_boxes = perform_detection(model, frame, confidence_threshold)
        else:
            # If it's not a frame for detection, use the original frame
            frame_with_boxes = frame

        # Display the original frame (or frame with bounding boxes)
        cv2.imshow("cam3", frame_with_boxes)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_no += 1

    # Release the video capture object and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()


def main(vid_path=None, vid_out=None):
    # Load YOLOv8 model
    model = YOLO('best.pt')
    model.to('cuda')


    cap = cv2.VideoCapture(vid_path)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Get video properties (frame width, height, and frames per second)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer (for saving filtered frames)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec for MP4
    out = cv2.VideoWriter(vid_out, fourcc, fps, (frame_width, frame_height))

    # Set confidence threshold (adjust as needed)
    confidence_threshold = 0.7




    frame_count = 0
    # Set skip factor (process every nth frame)
    skip_factor = 5 # Example: Process every 5th frame
    # Start a separate thread to display frames from the video source
    display_thread = threading.Thread(target=display_frames, args=(vid_path, model,  confidence_threshold))  # For webcam
    display_thread.start()




#main(vid_path='./dataGun.mp4', vid_out='outputt.avi')  ### for custom video
main(vid_path='rtsp://admin:kurtdecena1@192.168.1.11:554/mpeg4/ch1/main/av_stream', vid_out='live.avi')  ### for custom video