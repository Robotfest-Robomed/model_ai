import cv2
import time
import numpy as np
import onnxruntime as ort
import math
import os

# -----------------------------------
# CONFIG
# -----------------------------------
EMOTION_MODEL = "your_model.onnx"
POSE_MODEL = "model.onnx"
ALERT_IMAGE_PATH = "alert.png"

EMOTIONS = ["fear","neutral","happy","sad","anger","disgust","surprise","contempt"]
NEGATIVE_EMOTIONS = {"surprise","fear","disgust"}

EMOTION_ALERT_SEC = 1.0
HAND_NEAR_NECK_SEC = 1.0

POSE_INPUT_SIZE = 192
HAND_NEAR_THRESHOLD = 0.30

# -----------------------------------
# LOAD MODELS
# -----------------------------------
emotion_sess = ort.InferenceSession(EMOTION_MODEL, providers=["CPUExecutionProvider"])
emotion_in = emotion_sess.get_inputs()[0].name
emotion_out = emotion_sess.get_outputs()[0].name

pose_sess = ort.InferenceSession(POSE_MODEL, providers=["CPUExecutionProvider"])
pose_in = pose_sess.get_inputs()[0].name
pose_out = pose_sess.get_outputs()[0].name

# -----------------------------------
# ALERT IMAGE
# -----------------------------------
if os.path.exists(ALERT_IMAGE_PATH):
    alert_img = cv2.imread(ALERT_IMAGE_PATH)
else:
    alert_img = None
    print("⚠ WARNING: alert.png NOT FOUND!")

# -----------------------------------
# DRAW POSE
# -----------------------------------
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (0,5),(0,6),
    (5,7),(7,9),
    (6,8),(8,10),
    (5,6),
    (5,11),(6,12),
    (11,13),(13,15),
    (12,14),(14,16),
    (11,12)
]

def draw_pose(frame, kp):
    h, w, _ = frame.shape

    for (y, x, s) in kp:
        if s > 0.2:
            cv2.circle(frame, (int(x*w), int(y*h)), 3, (0,255,0), -1)

    for a,b in SKELETON:
        y1,x1,s1 = kp[a]
        y2,x2,s2 = kp[b]
        if s1>0.2 and s2>0.2:
            cv2.line(frame,
                     (int(x1*w),int(y1*h)),
                     (int(x2*w),int(y2*h)),
                     (0,255,255),2)
    return frame

# -----------------------------------
# HAND NEAR NECK RULE
# -----------------------------------
hand_near_start = None

def check_hand_near_neck(kp):
    global hand_near_start

    ls_y,ls_x,ls_s = kp[5]
    rs_y,rs_x,rs_s = kp[6]

    if ls_s<0.2 or rs_s<0.2:
        hand_near_start = None
        return False, None

    neck_x = (ls_x + rs_x) / 2
    neck_y = (ls_y + rs_y) / 2

    lw_y,lw_x,lw_s = kp[9]
    rw_y,rw_x,rw_s = kp[10]

    hands = []
    if lw_s > 0.2: hands.append((lw_x, lw_y))
    if rw_s > 0.2: hands.append((rw_x, rw_y))

    near = False
    for (hx,hy) in hands:
        d = math.sqrt((hx-neck_x)**2 + (hy-neck_y)**2)
        if d < HAND_NEAR_THRESHOLD:
            near = True

    now = time.time()

    if near:
        if hand_near_start is None:
            hand_near_start = now
        elif now - hand_near_start >= HAND_NEAR_NECK_SEC:
            return True, (neck_x, neck_y)
    else:
        hand_near_start = None

    return False, (neck_x, neck_y)

# -----------------------------------
# MAIN CAMERA LOOP
# -----------------------------------
cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

pose_mode = False
emotion_start = None
asthma_alert = False

print("✔ Emotion → Pose Asthma Detection Started")

prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------------
    # FPS CALCULATION
    # -------------------------
    now = time.time()
    dt = now - prev_time
    prev_time = now
    fps = 1 / dt if dt > 0 else 0

    # STOP ALL DETECTION IF ALERT TRIGGERED
    if asthma_alert:
        if alert_img is not None:
            alert_resized = cv2.resize(alert_img, (frame.shape[1], frame.shape[0]))
            frame = alert_resized

        cv2.putText(frame, "!!! ASTHMA ALERT !!!", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)

        cv2.imshow("ASTHMA AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue  # ← stops processing but keeps window open

    # ---------------------------------------
    # EMOTION DETECTION
    # ---------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    e_in = cv2.resize(gray, (64,64)).astype(np.float32) / 255.0
    e_in = np.expand_dims(e_in, (0,3))

    out = emotion_sess.run([emotion_out], {emotion_in: e_in})
    logits = out[0][0]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    e_idx = int(np.argmax(probs))
    emotion = EMOTIONS[e_idx]

    cv2.putText(frame, f"Emotion: {emotion}", (5,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # ---------------------------------------
    # ACTIVATE POSE MODE
    # ---------------------------------------
    if emotion.lower() in NEGATIVE_EMOTIONS:
        if emotion_start is None:
            emotion_start = now
        elif now - emotion_start >= EMOTION_ALERT_SEC:
            pose_mode = True
    else:
        emotion_start = None

    # ---------------------------------------
    # POSE MODE ACTIVE
    # ---------------------------------------
    if pose_mode:
        p_in = cv2.resize(frame, (POSE_INPUT_SIZE, POSE_INPUT_SIZE))
        p_in = cv2.cvtColor(p_in, cv2.COLOR_BGR2RGB)
        p_in = p_in.astype(np.int32)[np.newaxis, ...]

        out = pose_sess.run([pose_out], {pose_in: p_in})
        kp = out[0][0][0]

        frame = draw_pose(frame, kp)

        hand_alert, neck_point = check_hand_near_neck(kp)

        if neck_point is not None:
            H,W,_ = frame.shape
            cx,cy = int(neck_point[0]*W), int(neck_point[1]*H)
            cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)

        if hand_alert:
            asthma_alert = True

    # -------------------------
    # DRAW FPS
    # -------------------------
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-90,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0),1)

    cv2.imshow("ASTHMA AI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
