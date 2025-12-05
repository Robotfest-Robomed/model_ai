import cv2
import time
import numpy as np
import onnxruntime as ort
import os

# -----------------------
# CONFIG
# -----------------------
ONNX_MODEL_PATH = "your_model.onnx"  # path to your converted HDF5->ONNX model
EMOTIONS = ["neutral","happy","surprise","sad","anger","disgust","fear","contempt"]

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

NEGATIVE_EMOTIONS = {"surprise", "anger"}
ALERT_DURATION_SEC = 3  # seconds before alert triggers

ALERT_ICON_PATH = "alert.png"  # optional transparent PNG

# -----------------------
# LOAD ONNX MODEL
# -----------------------
sess = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# -----------------------
# HELPER TO OVERLAY TRANSPARENT IMAGE
# -----------------------
def overlay_image(frame, img, x, y):
    """Overlay img (BGRA) onto frame at position x,y"""
    if img is None:
        return frame
    h, w = img.shape[:2]
    if y < 0 or x < 0 or y+h > frame.shape[0] or x+w > frame.shape[1]:
        return frame
    alpha = img[:, :, 3] / 255.0
    for c in range(3):
        frame[y:y+h, x:x+w, c] = (alpha * img[:, :, c] + (1 - alpha) * frame[y:y+h, x:x+w, c])
    return frame

# Load alert icon if exists
alert_icon = None
if os.path.exists(ALERT_ICON_PATH):
    alert_icon = cv2.imread(ALERT_ICON_PATH, cv2.IMREAD_UNCHANGED)
    if alert_icon is None:
        print("⚠ Failed to load alert icon, continuing without it")
else:
    print("⚠ Alert icon not found, continuing without it")

# -----------------------
# CAMERA SETUP
# -----------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

print("✔ Camera started. Press 'q' to quit.")

prev_time = time.time()
alert_start_time = None
alert_triggered = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_input = cv2.resize(gray, (64,64)).astype(np.float32)/255.0
        face_input = np.expand_dims(face_input, axis=(0,3))  # [1,64,64,1]

        # ONNX inference
        outputs = sess.run([output_name], {input_name: face_input})
        logits = outputs[0][0]
        probs = np.exp(logits)/np.sum(np.exp(logits))
        emotion_idx = int(np.argmax(probs))
        emotion_label = EMOTIONS[emotion_idx]

        # FPS
        now = time.time()
        fps = 1/(now-prev_time)
        prev_time = now

        # Print emotion and FPS
        print(f"Emotion: {emotion_label:<8} | FPS: {fps:.1f}")

        # Check for negative emotions
        if not alert_triggered and emotion_label.lower() in NEGATIVE_EMOTIONS:
            if alert_start_time is None:
                alert_start_time = now
            elif now - alert_start_time >= ALERT_DURATION_SEC:
                alert_triggered = True
        else:
            alert_start_time = None  # reset if neutral/happy/etc

        # Show alert if triggered (no white overlay)
        if alert_triggered:
            # Draw icon in center
            if alert_icon is not None:
                ih, iw = alert_icon.shape[:2]
                scale = 0.3  # 30% of original size
                icon_resized = cv2.resize(alert_icon, (int(iw*scale), int(ih*scale)))
                ih, iw = icon_resized.shape[:2]
                frame = overlay_image(frame, icon_resized,
                                      x=CAMERA_WIDTH//2 - iw//2,
                                      y=CAMERA_HEIGHT//2 - ih//2 - 10)

            # Draw text slightly below icon
            title = "ASTHMA ALERT!"
            subtitle = "Negative Emotion Detected"
            font_scale_title = 0.35 * CAMERA_WIDTH / 320
            font_scale_sub = 0.2 * CAMERA_WIDTH / 320
            thickness_title = 1
            thickness_sub = 1

            (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, font_scale_title, thickness_title)
            (sw, sh), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_DUPLEX, font_scale_sub, thickness_sub)

            frame = cv2.putText(frame, title, (CAMERA_WIDTH//2 - tw//2, CAMERA_HEIGHT//2 + ih//2 + 10),
                                cv2.FONT_HERSHEY_DUPLEX, font_scale_title, (0,0,255), thickness_title)
            frame = cv2.putText(frame, subtitle, (CAMERA_WIDTH//2 - sw//2, CAMERA_HEIGHT//2 + ih//2 + 25),
                                cv2.FONT_HERSHEY_DUPLEX, font_scale_sub, (0,0,255), thickness_sub)

        # Show camera feed
        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
