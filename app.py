from flask import Flask, Response, jsonify, send_from_directory
import cv2
import os
import sqlite3
import threading
from datetime import datetime
from cam6 import perform_detection  # Import detection logic

app = Flask(__name__)

# Database setup
DB_PATH = "mydb.db"
TABLE_NAME = "detections"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            label TEXT,
            confidence REAL,
            image_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Image storage path
SAVE_DIR = "detected_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

def generate_frames():
    """Continuously captures frames from the webcam and processes them for object detection."""
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Perform object detection
        frame, detections = perform_detection(frame)
        
        # Save detected images and log them in the database
        for (x1, y1, x2, y2, conf, label) in detections:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"{label}_{timestamp}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            
            cv2.imwrite(filepath, frame)
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(f'''
                INSERT INTO {TABLE_NAME} (timestamp, label, confidence, image_path)
                VALUES (?, ?, ?, ?)
            ''', (timestamp, label, conf, filepath))
            conn.commit()
            conn.close()

        # Encode frame to stream
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Streams real-time video with detected objects."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_images', methods=['GET'])
def get_images():
    """Returns a list of saved detection images."""
    images = sorted(
        [img for img in os.listdir(SAVE_DIR) if img.endswith(('jpg', 'png', 'jpeg'))],
        reverse=True
    )
    return jsonify({"images": images})

@app.route('/detected_images/<filename>')
def serve_image(filename):
    """Serves detected images."""
    return send_from_directory(SAVE_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
