# People Counter with YOLOv8

This project implements an automatic pedestrian counting system using
computer vision and deep learning.

It detects and tracks people in real time and counts how many pedestrians
cross a virtual line in each direction.

The system is designed for testing on pedestrian crossings and supports
both PC and Raspberry Pi deployment.

---

## Features

- Real-time person detection using YOLOv8
- Multi-object tracking with persistent IDs
- Gate-based virtual counting zone for higher accuracy
- Left-to-right and right-to-left direction counting
- CSV logging of counting events
- Optional video recording (raw + overlay)

---

## Technologies

- Python  
- OpenCV  
- Ultralytics YOLOv8  

---

## Files

- `people_counter_camera.py` – basic camera-based counter  
- `people_counter_gate.py` – improved gate + tracking version  
- `yolov8n.pt` – YOLOv8 Nano model weights  
- `.gitignore` – excludes generated videos and logs  

---

## Usage

### 1. Install dependencies

```bash
pip install ultralytics opencv-python numpy
```
### 2. Connect a camera or use a video file

- Camera (default): the script uses `SOURCE = 0`
- Video file: set `SOURCE = "path/to/video.mp4"`

### 3. Run the counter

Basic version:

```bash
python people_counter_camera.py
```
Improved gate + tracking version:
```bash
python people_counter_gate.py
```

---

## Configuration

Main parameters that affect accuracy (in `people_counter_gate.py`):

- `GATE_HALF_WIDTH` – width of the counting zone around the line
- `MIN_TRACK_AGE` – minimum frames before counting a track
- `COUNT_COOLDOWN_SEC` – prevents double counting due to jitter
- `MIN_X_DISPLACEMENT` – minimum x-movement required for a valid crossing

You may also need to adjust the line position depending on the camera angle:

- Default: `line_x = frame_width // 2`
- Manual example: `line_x = int(frame_width * 0.45)`

---

## Notes (PC vs Raspberry Pi)

- On a stronger PC you can usually run at higher FPS and/or higher resolution,
  which improves tracking stability and counting accuracy.
- On Raspberry Pi, performance is lower, so recommended improvements are:
  - lower resolution (e.g. 640x480)
  - use `yolov8n.pt`
  - tune `GATE_HALF_WIDTH` and `COUNT_COOLDOWN_SEC` for more stable counts

---

## Output

Depending on configuration, the script can produce:

- `people_count_log.csv` – timestamps and directional counts
- `raw.avi` / `raw.mp4` – raw recording (no overlay)
- `overlay.avi` / `overlay.mp4` – overlay recording (boxes, line, counters)

Note: generated videos/logs are excluded from Git using `.gitignore`.

---

## Limitations

- Accuracy depends on camera quality and frame rate
- Heavy occlusions can cause ID switching (tracking errors)
- The counting line position must be calibrated to the scene

---

## Project Context

This project was developed as part of a student project on automatic pedestrian counting.

Repository:
https://github.com/Naks00/People-counter-YOLOv8

---

## Demo Video

A short demonstration and testing video of the system is available on YouTube:

https://www.youtube.com/watch?v=7CUsR9kc0gc
