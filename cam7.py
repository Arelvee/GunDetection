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

from flask import Flask, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)

pygame.mixer.init()
ALARM_SOUND = "notif.mp3"

app = Flask(__name__)
CORS(app) 

SAVE_DIR = os.path.join(app.root_path, 'static/detected_images')
os.makedirs(SAVE_DIR, exist_ok=True)

DB_PATH = "detection_history.db"

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

def capture_frames():
    """ Continuously captures frames from the camera without interruptions. """
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
    """ Saves detected images asynchronously without interrupting the stream. """
    filename = f"detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)
    
    success = cv2.imwrite(filepath, frame)
    if success:
        logging.info(f"✅ Auto-saved detected image: {os.path.abspath(filepath)}")
    else:
        logging.error("❌ Failed to auto-save image.")

def save_detection_to_db(label, confidence):
    """ Saves detection data into SQLite database """
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
    """ Streams frames and saves detected images asynchronously without lag. """
    while True:
        with frame_lock:
            if latest_frame is None:
                continue

            frame = latest_frame.copy()

        try:
            results = model(frame)
            if results and len(results[0].boxes) > 0:
                detections = sv.Detections.from_ultralytics(results[0])
                labels = []
                
                for cls, conf in zip(results[0].boxes.cls, results[0].boxes.conf):
                    label = f"{model.names[int(cls)]}"
                    confidence = float(conf)
                    labels.append(f"{label} (Confidence: {confidence:.2%})")
                    save_detection_to_db(label, confidence)  # Save to database
                
                annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

                save_queue.put(annotated_frame.copy())  # ✅ Add to queue without blocking
                
                if any("unregistered firearm" in label for label in labels):
                    pygame.mixer.music.load(ALARM_SOUND)
                    pygame.mixer.music.play()

                display_frame = annotated_frame
            else:
                display_frame = frame

            # ✅ Keep the stream alive without interruptions
            _, buffer = cv2.imencode('.jpg', display_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            logging.error(f"Error processing frame: {e}")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_images', methods=['GET'])
def get_images():
    images = [img for img in os.listdir(SAVE_DIR) if img.endswith(('jpg', 'png', 'jpeg'))]
    logging.info(f"Available images: {images}")
    return jsonify({"images": images})

@app.route('/detected_images/<filename>')
def serve_image(filename):
    return send_from_directory(SAVE_DIR, filename)

# Function to retrieve detection history from the database
def get_detection_history():
    conn = sqlite3.connect('detection_history.db')
    cursor = conn.cursor()
    cursor.execute("SELECT label, confidence, timestamp FROM detections ORDER BY timestamp DESC")
    history = cursor.fetchall()
    conn.close()
    
    # Format the results as a list of dictionaries
    history_data = []
    for entry in history:
        history_data.append({
            'label': entry[0],
            'confidence': entry[1],
            'timestamp': entry[2]
        })
    return history_data

@app.route('/detection_history', methods=['GET'])
def detection_history():
    try:
        history = get_detection_history()
        if history:
            return jsonify(history)
        else:
            return jsonify([])  # Empty list if no history
    except Exception as e:
        print(f"Error fetching history: {e}")
        return jsonify({"error": "Failed to load history"}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)