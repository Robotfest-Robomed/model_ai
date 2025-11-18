import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import mediapipe as mp
from collections import Counter

# ----------------------------
# Paths
# ----------------------------
input_folder = "input_images"   # folder with your images
output_folder = "output_images" # folder to save results
os.makedirs(output_folder, exist_ok=True)

# ----------------------------
# Load TFLite model
# ----------------------------
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model input shape:", input_details[0]['shape'])

# ----------------------------
# Emotion labels
# ----------------------------
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# ----------------------------
# Setup Mediapipe Face Detection
# ----------------------------
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# ----------------------------
# Counter for all emotions
# ----------------------------
emotion_counter = Counter()
total_faces = 0

# ----------------------------
# Process each image
# ----------------------------
for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(input_folder, filename)
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Could not read {filename}")
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)

    if results.detections:
        for detection in results.detections:
            total_faces += 1

            # Bounding box
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            x, y = max(0, x), max(0, y)
            w, h = min(iw - x, w), min(ih - y, h)

            # Crop and preprocess
            face_color = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face_color, (64, 64))
            face_input = face_resized.astype("float32") / 255.0
            face_input = np.expand_dims(face_input, axis=0)  # shape (1,64,64,3)

            # Run inference
            interpreter.set_tensor(input_details[0]['index'], face_input)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]

            # Get predicted emotion
            emotion = EMOTIONS[np.argmax(preds)]
            confidence = np.max(preds)
            emotion_counter[emotion] += 1

            # Draw bounding box + label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    else:
        print(f"No face detected in {filename}")

    # Save output image
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, frame)
    print(f"Processed {filename} → saved to {output_path}")

# ----------------------------
# Print emotion statistics
# ----------------------------
print("\n=== Emotion Statistics ===")
if total_faces == 0:
    print("No faces detected in any image.")
else:
    for emotion in EMOTIONS:
        count = emotion_counter[emotion]
        percentage = (count / total_faces) * 100
        print(f"{emotion}: {count} faces ({percentage:.2f}%)")
