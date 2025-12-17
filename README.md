# People Counter with YOLOv8

This repository contains a prototype people counting system based on a USB camera and a YOLOv8 object detection model. The goal is to detect pedestrians crossing a virtual line and log their movement over time for later analysis.

## Features

- Real-time person detection using YOLOv8 (`yolov8n.pt`)
- Virtual counting line with left-to-right / right-to-left direction detection
- Logging of all events to a CSV file (`people_count_log.csv`)
- Optional video recording of the processed stream with overlays (`people_count_output.avi`)

## Files

- `people_counter_camera.py` – main script for camera capture, YOLOv8 inference and counting logic  
- `people_count_log.csv` – example log file with timestamps and counts  
- `people_count_output.avi` – example output video with detections and counters  
- `yolov8n.pt` – YOLOv8 Nano model weights used for person detection  

## Usage

1. Install Python dependencies (e.g. `ultralytics`, `opencv-python`, `numpy`).  
2. Connect a camera supported by OpenCV.  
3. Run: python people_counter_camera.py
4. The script will open a window with the video feed, draw detections and update the counters.

This code was developed as part of a student project on automatic pedestrian counting, more here https://github.com/Naks00/People-counter 

