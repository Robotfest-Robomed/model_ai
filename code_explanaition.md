This script performs emotion recognition on all images inside a folder using a TensorFlow Lite model and MediaPipe face detection.
For each detected face, it draws a bounding box with the predicted emotion, saves the annotated output image, and counts how often each emotion appears.
After processing all images, it prints a summary of emotion statistics and percentages across the entire dataset.
