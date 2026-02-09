import cv2
from ultralytics import YOLO
import csv
import time


model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)  
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20  


fourcc = cv2.VideoWriter_fourcc(*'XVID')  # codec (može i 'MJPG', 'mp4v')
out = cv2.VideoWriter('people_count_output.avi', fourcc, fps, (frame_width, frame_height))

line_x = frame_width // 2  # pozicija vertikalne linije u sredini slike

people_up = 0
people_down = 0
previous_centers = []

file = open('people_count_log.csv', mode='a', newline='')
writer = csv.writer(file)
last_people_up = 0
last_people_down = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    persons = [box for box in results.boxes if int(box.cls) == 0]

    current_centers = []
    for person in persons:
        x1, y1, x2, y2 = map(int, person.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        current_centers.append((cx, cy))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # Provjera prelaska vertikalne linije
    for pc in previous_centers:
        for cc in current_centers:
            if abs(pc[1] - cc[1]) < 50:  # pripadaju istom visinskom području (y koordinata)
                if pc[0] < line_x <= cc[0]:  # prelaz s lijeva na desno
                    people_down += 1
                elif pc[0] > line_x >= cc[0]:  # prelaz s desna na lijevo
                    people_up += 1

    previous_centers = current_centers.copy()

    
    if people_up != last_people_up or people_down != last_people_down:
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), people_up, people_down])
        last_people_up = people_up
        last_people_down = people_down

 
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 0, 0), 2)
# tekst s lijeve strane linije
    left_text = f"Left to Right: {people_down}"
    cv2.putText(frame, left_text, (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# tekst s desne strane linije
    right_text = f"Right to Left: {people_up}"
    text_size, _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_width = text_size[0]
    cv2.putText(frame, right_text, (frame_width - text_width - 10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


   
    out.write(frame)

    cv2.imshow('People Counter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

file.close()
cap.release()
out.release()
cv2.destroyAllWindows()
