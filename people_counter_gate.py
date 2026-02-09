import cv2
import time
import csv
from collections import deque
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
SOURCE = "raw4 - Trim.mp4"

FOURCC = "XVID"
FPS_FALLBACK = 20

RAW_OUT = "raw.avi"
OVERLAY_OUT = "overlay.avi"
CSV_LOG = "people_count_log.csv"

line_x = None

# ----- TUNING ZA GUŽVU -----
GATE_HALF_WIDTH = 80        # šira zona → lakše “uhvati” prelaz
MIN_TRACK_AGE = 4           # brže dopušta brojanje
MIN_X_DISPLACEMENT = 15     # manji pomak dovoljan
COUNT_COOLDOWN_SEC = 0.6    # kraći cooldown
STALE_ID_SEC = 3.5
# ---------------------------

def side(cx, x0):
    return "L" if cx < x0 else "R"

def in_gate(cx, x0, half_w):
    return abs(cx - x0) <= half_w

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(SOURCE)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 1:
    fps = FPS_FALLBACK

line_x = w // 2
fourcc = cv2.VideoWriter_fourcc(*FOURCC)
out_raw = cv2.VideoWriter(RAW_OUT, fourcc, fps, (w, h))
out_vis = cv2.VideoWriter(OVERLAY_OUT, fourcc, fps, (w, h))

# state per ID
pos_hist = {}        # tid -> deque([(cx,cy), ...])
track_age = {}       # tid -> frames seen
last_seen = {}       # tid -> ts
last_count = {}      # tid -> ts
entered_gate = {}    # tid -> bool
entry_side = {}      # tid -> "L"/"R" (strana kad je ušao u gate)

count_L2R = 0
count_R2L = 0

file = open(CSV_LOG, "a", newline="")
writer = csv.writer(file)

last_log = (-1, -1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    raw = frame.copy()
    vis = frame.copy()


    results = model.track(vis, persist=True, classes=[0], verbose=False)[0]

    # crtaj zonu (vizualno)
    cv2.line(vis, (line_x, 0), (line_x, h), (255, 0, 0), 2)
    #cv2.line(vis, (line_x - GATE_HALF_WIDTH, 0), (line_x - GATE_HALF_WIDTH, h), (255, 0, 0), 1)
    #cv2.line(vis, (line_x + GATE_HALF_WIDTH, 0), (line_x + GATE_HALF_WIDTH, h), (255, 0, 0), 1)

    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), tid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            last_seen[tid] = now
            track_age[tid] = track_age.get(tid, 0) + 1
            last_count.setdefault(tid, 0.0)

            if tid not in pos_hist:
                pos_hist[tid] = deque(maxlen=6)
            pos_hist[tid].append((cx, cy))

            # overlay
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(vis, f"ID:{tid}", (x1, max(0, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # trebamo barem 2 točke za putanju
            if len(pos_hist[tid]) < 2:
                continue

            if track_age[tid] < MIN_TRACK_AGE:
                continue

            (px, py) = pos_hist[tid][-2]
            (nx, ny) = pos_hist[tid][-1]

            # 1) detektiraj ulazak u gate zonu
            gate_now = in_gate(nx, line_x, GATE_HALF_WIDTH)
            if gate_now and not entered_gate.get(tid, False):
                entered_gate[tid] = True
                entry_side[tid] = side(nx, line_x)

            # 2) ako je ušao u gate, čekaj da izađe na drugu stranu i da je pomak realan
            if entered_gate.get(tid, False):
                cooldown_ok = (now - last_count[tid]) >= COUNT_COOLDOWN_SEC

                
                if not gate_now and cooldown_ok:
                    exit_side = side(nx, line_x)
                    ent_side = entry_side.get(tid)

                    # mora biti druga strana od ulaza
                    if ent_side is not None and exit_side != ent_side:
                       
                        xs = [p[0] for p in pos_hist[tid]]
                        dx = max(xs) - min(xs)

                        if dx >= MIN_X_DISPLACEMENT:
                            if ent_side == "L" and exit_side == "R":
                                count_L2R += 1
                            elif ent_side == "R" and exit_side == "L":
                                count_R2L += 1

                            last_count[tid] = now

                    # reset: mora opet ući u gate za novo brojanje
                    entered_gate[tid] = False
                    entry_side.pop(tid, None)

    # cleanup starih ID-eva
    stale = [tid for tid, ts in last_seen.items() if (now - ts) > STALE_ID_SEC]
    for tid in stale:
        last_seen.pop(tid, None)
        last_count.pop(tid, None)
        track_age.pop(tid, None)
        entered_gate.pop(tid, None)
        entry_side.pop(tid, None)
        pos_hist.pop(tid, None)

    # Lijeva strana – Left to Right
    cv2.putText(vis, f"Left to Right: {count_L2R}", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Desna strana – Right to Left
    text = f"Right to Left: {count_R2L}"
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
    )

    x_pos = w - text_w - 10  # 10 px margina od ruba

    cv2.putText(vis, text, (x_pos, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    if (count_L2R, count_R2L) != last_log:
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), count_L2R, count_R2L])
        last_log = (count_L2R, count_R2L)

    out_raw.write(raw)   # čisti video
    out_vis.write(vis)   # overlay video

    cv2.imshow("People Counter (overlay)", vis)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

file.close()
cap.release()
out_raw.release()
out_vis.release()
cv2.destroyAllWindows()
