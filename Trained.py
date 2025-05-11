from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/dataset')
#pwd

#!pip install torch
import torch
print(torch.__version__)
from IPython.display import Image
#! pip install  -r /content/drive/MyDrive/dataset/requirements.txt

#!pip install ultralytics
%git clone https://github.com/ultralytics/yolov5
%cd yolov5

import os
from random import choice
import shutil

train_imgs = []
train_labels = []
val_imgs = []
val_labels = []

trainimagePath = '/content/drive/MyDrive/dataset/images/train'
trainlabelPath = '/content/drive/MyDrive/dataset/labels/train'
valimagePath = '/content/drive/MyDrive/dataset/images/val'
vallabelPath = '/content/drive/MyDrive/dataset/labels/val'
for (dirname, dirs, files) in os.walk(trainimagePath):
    for filename in files:
        if filename.endswith('.jpg'):
            train_imgs.append(filename)

for (dirname, dirs, files) in os.walk(trainlabelPath):
    for filename in files:
        if filename.endswith('.txt'):
            train_labels.append(filename[:-4] + '.jpg')
train_imgs = [img for img in train_imgs if img in train_labels]

for _ in range(num_val_images):
    fileJpg = choice(train_imgs)  
    fileTxt = fileJpg[:-4] + '.txt' 

    shutil.move(os.path.join(trainimagePath, fileJpg), os.path.join(valimagePath, fileJpg))
    shutil.move(os.path.join(trainlabelPath, fileTxt), os.path.join(vallabelPath, fileTxt))

    train_imgs.remove(fileJpg)
    train_labels.remove(fileJpg)

val_labels = [f for f in os.listdir(vallabelPath) if f.endswith('.txt')]

if not val_labels:
    print("WARNING ⚠️ no labels found in val set, can not compute metrics without labels")
else:
    print(f"Validation set contains {len(val_labels)} label files.")
!python /content/drive/MyDrive/dataset/yolov5/train.py --img-size 500 --batch-size 10 --epochs 50 --data /content/drive/MyDrive/dataset/data.yaml --cfg /content/drive/MyDrive/dataset/yolov5/models/yolov5s.yaml --weights /content/drive/MyDrive/dataset

from IPython.display import Image

image_path = "/content/drive/MyDrive/dataset/yolov5/runs/train/exp8/train_batch0.jpg"
Image(filename=image_path)
image_path = "/content/drive/MyDrive/dataset/yolov5/runs/train/exp8/train_batch2.jpg"
Image(filename=image_path)

import matplotlib.pyplot as plt

results_path = '/content/drive/MyDrive/dataset/yolov5/runs/train/exp8'  # Adjust the path if necessary
def plot_results(results_path):
    results_file = os.path.join(results_path, 'results.png')
    if os.path.exists(results_file):
        img = plt.imread(results_file)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        print(f"No results found at {results_file}")
plot_results(results_path)