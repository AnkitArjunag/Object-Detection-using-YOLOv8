from flask import Flask, request, render_template, send_file
from ultralytics import YOLO
import cv2
import torch
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
app = Flask(__name__)

model = YOLO('yolov8m.pt')
target_classes = ['person', 'car']
confidence_threshold = 0.6

def get_class_name(class_id):
    class_names = model.names  
    return class_names[class_id]

def save_annotated_image(image_path, output_path):
    # ---- ABSOLUTE PATH ----
    image_path = os.path.abspath(image_path)

    # ---- VALIDATE FILE ----
    if not os.path.exists(image_path):
        raise ValueError(f"File does not exist: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("OpenCV failed to load image. Unsupported or corrupted file.")

    # ---- RUN YOLO ON IMAGE ARRAY (BEST PRACTICE) ----
    results = model(img)

    target_classes = ['person', 'car', 'motorcycle', 'bus', 'truck']
    confidence_threshold = 0.6

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            if class_name in target_classes and confidence >= confidence_threshold:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    cv2.imwrite(output_path, img)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        file_path = os.path.join('uploads', file.filename)
        output_path = os.path.join('output', 'annotated_' + file.filename)
        file.save(file_path)
        save_annotated_image(file_path, output_path)
        return send_file(output_path, mimetype='image/jpeg')
 
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    app.run(debug=True)
