from flask import Flask, request, render_template, send_file
from ultralytics import YOLO
import cv2
import torch
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
app = Flask(__name__)

model = YOLO('yolov8m.pt')
target_class = 'person' 
confidence_threshold = 0.3

def get_class_name(class_id):
    class_names = model.names  
    return class_names[class_id]

def save_annotated_image(image_path, output_path):
    results = model(image_path)
    img = cv2.imread(image_path)

    for result in results:
        for box in result.boxes:
            if isinstance(box.xyxy, torch.Tensor):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0].item())
                confidence = box.conf[0].item()
            else:
                x1, y1, x2, y2 = box.xyxy
                class_id = int(box.cls)
                confidence = box.conf

            class_name = get_class_name(class_id)
            if class_name == target_class and confidence >= confidence_threshold:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
                label = f"{class_name} {confidence:.2f}"
                font_path = ".fonts/calibri.ttf"
                font_size = 14
                calibri_font = ImageFont.truetype(font_path, font_size)
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                draw.text((int(x1), int(y1) - 10), label, font=calibri_font, fill=(0, 0, 0))
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
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
