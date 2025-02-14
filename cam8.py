import os
import cv2
import logging
import threading
import pygame
import sqlite3
from datetime import datetime
from flask import Flask, Response, jsonify, send_from_directory
from queue import Queue

from ultralytics import YOLO
import supervision as sv
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)

pygame.mixer.init()
ALARM_SOUND = "notif.mp3"

app = Flask(__name__)
CORS(app)

SAVE_DIR = os.path.join(app.root_path, 'static/detected_images')
os.makedirs(SAVE_DIR, exist_ok=True)

DB_PATH = "detection_history.db"

# Initialize database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            label TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    logging.error(f"Model file '{MODEL_PATH}' not found!")
    exit(1)

try:
    model = YOLO(MODEL_PATH)
    logging.info(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit(1)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    logging.error("Camera failed to open. Check if the camera is connected or in use.")
    exit(1)

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

frame_lock = threading.Lock()
latest_frame = None
save_queue = Queue()

KNOWN_OFFICERS = ["PO Kim Eugene Buena", "SO Ruby Mae Medequillo"]

def capture_frames():
    """ Continuously captures frames from the camera. """
    global latest_frame
    while True:
        success, frame = camera.read()
        if not success:
            logging.error("Failed to read from camera.")
            break

        with frame_lock:
            latest_frame = frame.copy()

thread = threading.Thread(target=capture_frames, daemon=True)
thread.start()

def save_detected_image(frame):
    """ Saves detected images asynchronously. """
    filename = f"detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)
    success = cv2.imwrite(filepath, frame)
    if success:
        logging.info(f"✅ Image saved: {os.path.abspath(filepath)}")
    else:
        logging.error("❌ Failed to save image.")

def save_detection_to_db(label, confidence):
    """ Saves detection data into SQLite database. """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO detections (timestamp, label, confidence) VALUES (?, ?, ?)", 
                   (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), label, confidence))
    conn.commit()
    conn.close()

def save_worker():
    """ Background thread that continuously saves detected images. """
    while True:
        frame = save_queue.get()
        save_detected_image(frame)
        save_queue.task_done()

save_thread = threading.Thread(target=save_worker, daemon=True)
save_thread.start()

def generate_frames():
    """ Streams frames with firearm and face recognition security alerts. """
    while True:
        with frame_lock:
            if latest_frame is None:
                continue

            frame = latest_frame.copy()

        try:
            results = model(frame)
            detected_objects = []
            detected_faces = []

            if results and len(results[0].boxes) > 0:
                detections = sv.Detections.from_ultralytics(results[0])

                for cls, conf in zip(results[0].boxes.cls, results[0].boxes.conf):
                    label = f"{model.names[int(cls)]}"
                    confidence = float(conf)

                    if label in ["rifle", "handgun", "shotgun", "revolver", "unregistered firearm", "bladed weapon"]:
                        detected_objects.append(label)
                    elif label in KNOWN_OFFICERS:
                        detected_faces.append(label)

                    save_detection_to_db(label, confidence)

                # 🚨 SECURITY ALERT SYSTEM 🚨
                alert_message = "No Threat Detected"
                alert_color = (0, 255, 0)  # Green (Safe)
                play_sound = False

                if any(obj in ["rifle", "handgun", "shotgun", "revolver"] for obj in detected_objects):
                    if detected_faces:
                        alert_message = "SAFE"
                        alert_color = (0, 255, 0)  # Green
                    else:
                        alert_message = "WARNING"
                        alert_color = (255, 215, 0)  # Yellow Gold
                        play_sound = True

                elif any(obj in ["unregistered firearm", "bladed weapon"] for obj in detected_objects):
                    if detected_faces:
                        alert_message = "WARNING"
                        alert_color = (255, 215, 0)  # Yellow Gold
                        play_sound = True
                    else:
                        alert_message = "THREAT!"
                        alert_color = (255, 0, 0)  # Red
                        play_sound = True

                # Play Alarm Sound
                if play_sound:
                    pygame.mixer.music.load(ALARM_SOUND)
                    pygame.mixer.music.play()

                # Draw Alert Message
                cv2.putText(frame, alert_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, alert_color, 2)

                # Annotate and save frame
                annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

                save_queue.put(annotated_frame.copy())
                display_frame = annotated_frame
            else:
                display_frame = frame

            # Stream Frame
            _, buffer = cv2.imencode('.jpg', display_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            logging.error(f"Error processing frame: {e}")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Serve images from the static directory
@app.route('/get_images')
def get_images():
    image_folder = "static/detected_images"
    try:
        images = os.listdir(image_folder)  # List all images
        return jsonify({"images": images})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve individual images
@app.route('/detected_images/<filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(app.static_folder, 'detected_images'), filename)

def get_detection_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT label, confidence, timestamp FROM detections ORDER BY timestamp DESC")
    history = cursor.fetchall()
    conn.close()

    return [{"label": entry[0], "confidence": entry[1], "timestamp": entry[2]} for entry in history]

@app.route('/detection_history', methods=['GET'])
def detection_history():
    try:
        history = get_detection_history()
        return jsonify(history if history else [])
    except Exception as e:
        logging.error(f"Error fetching history: {e}")
        return jsonify({"error": "Failed to load history"}), 500
    

@app.route('/get_alert_status', methods=['GET'])
def get_alert_status():
    # Capture a frame from the camera
    ret, frame = camera.read()
    if not ret:
        return jsonify({"error": "Unable to read camera frame"}), 500

    # Run YOLO detection
    results = model(frame)
    detected_objects = []

    if results and len(results[0].boxes) > 0:
        for cls in results[0].boxes.cls:
            label = model.names[int(cls)]
            detected_objects.append(label)

    # 🚨 SECURITY ALERT SYSTEM 🚨
    alert_message = "No Threat Detected"
    alert_color = (0, 255, 0)  # Green (Safe)
    play_sound = False
    detected_faces = any(obj in KNOWN_OFFICERS for obj in detected_objects)  # Check if a known officer is detected

    # Check for firearm or weapon detection
    if any(obj in ["rifle", "handgun", "shotgun", "revolver"] for obj in detected_objects):
        if detected_faces:
            alert_message = "SAFE"
        else:
            alert_message = "WARNING"
            alert_color = (255, 215, 0)  # Yellow Gold
            play_sound = True

    elif any(obj in ["unregistered firearm", "bladed weapon"] for obj in detected_objects):
        if detected_faces:
            alert_message = "WARNING"
            alert_color = (255, 215, 0)  # Yellow Gold
            play_sound = True
        else:
            alert_message = "THREAT!"
            alert_color = (255, 0, 0)  # Red
            play_sound = True

    return jsonify({
        "alert_message": alert_message,
        "alert_color": list(alert_color),  # Convert tuple to list for JSON serialization
        "play_sound": play_sound
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
