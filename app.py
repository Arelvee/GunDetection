from flask import Flask, Response, jsonify, send_from_directory, render_template
import cv2
import os
import sqlite3
from datetime import datetime
from cam6 import perform_detection  # Import detection logic

app = Flask(__name__)

# Database setup
DB_PATH = "mydb.db"
TABLE_NAME = "detections"

def init_db():
    """Initialize the database and create the table if it doesn't exist."""
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
    global cap  # Ensure cap is accessible

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break  # Exit loop if frame capture fails

        # Perform object detection
        CONFIDENCE_THRESHOLD = 0.6
        frame, detections = perform_detection(frame, CONFIDENCE_THRESHOLD)

        # Save detected images and log them in the database
        for (x1, y1, x2, y2, conf, label) in detections:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"{label}_{timestamp}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)

            # Crop the detected object before saving
            cropped_object = frame[y1:y2, x1:x2]
            cv2.imwrite(filepath, cropped_object)

            # Insert detection record into the database
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                    INSERT INTO {TABLE_NAME} (timestamp, label, confidence, image_path)
                    VALUES (?, ?, ?, ?)
                ''', (timestamp, label, conf, filepath))
                conn.commit()

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    """Serve the HTML page with the live camera feed."""
    return render_template("dashboard.html")

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
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
