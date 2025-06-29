import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Load model
model = MobileNetV2(weights='imagenet')

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

frame_count = 0
prediction_interval = 10
last_label = "Loading..."

# Fun BGR colors to cycle through
colors = [
    (255, 0, 255),    # Purple
    (0, 165, 255),    # Orange
    (255, 255, 0),    # Cyan
    (0, 255, 255),    # Yellow
    (203, 192, 255),  # Pink
    (255, 0, 0),      # Blue
    (0, 0, 255),      # Red
]

color_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % prediction_interval == 0:
        img = cv2.resize(frame, (224, 224))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = preprocess_input(np.expand_dims(img_rgb, axis=0))

        preds = model.predict(x, verbose=0)
        decoded = decode_predictions(preds, top=1)[0][0]
        last_label = f"{decoded[1]}: {decoded[2]*100:.2f}%"

        color_index = (color_index + 1) % len(colors)

    # Draw label with the current color
    cv2.putText(frame, last_label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, colors[color_index], 2)

    cv2.imshow('Object Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
