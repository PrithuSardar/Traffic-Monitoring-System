
!pip install -q ultralytics opencv-python-headless

!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt

import cv2
import os
from matplotlib import pyplot as plt
from IPython.display import Video, display
from yolov5 import train, val, detect
import torch

!mkdir -p ../ua-detrac
%cd ../ua-detrac
!wget -c https://detrac-db.rit.albany.edu/Data/Insight-MVT_Annotation_Test.zip
!unzip -q Insight-MVT_Annotation_Test.zip

video_path = 'Insight-MVT_Annotation_Test/MVI_40731.avi'
frames_dir = 'frames'

os.makedirs(frames_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_filename = os.path.join(frames_dir, f"frame_{frame_id:04d}.jpg")
    cv2.imwrite(frame_filename, frame)
    frame_id += 1
cap.release()

print(f"Extracted {frame_id} frames.")

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

output_dir = 'output_frames'
os.makedirs(output_dir, exist_ok=True)

frame_files = sorted(os.listdir(frames_dir))

for frame_file in frame_files:
    img_path = os.path.join(frames_dir, frame_file)
    img = cv2.imread(img_path)
    results = model(img)
    results.render()  

    
    output_path = os.path.join(output_dir, frame_file)
    cv2.imwrite(output_path, results.ims[0])

print("Detection completed on frames.")

output_video_path = 'detected_traffic.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame = cv2.imread(os.path.join(output_dir, frame_files[0]))
height, width, _ = frame.shape
out = cv2.VideoWriter(output_video_path, fourcc, 15, (width, height))

for frame_file in frame_files:
    frame = cv2.imread(os.path.join(output_dir, frame_file))
    out.write(frame)

out.release()

display(Video(output_video_path, embed=True))
