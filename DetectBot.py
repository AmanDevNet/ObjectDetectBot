import cv2
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Initialize webcam
cap = cv2.VideoCapture(0)

frame_count = 0  # Counter to limit the number of frames
max_frames = 100  # Stop automatically after processing 100 frames
start_time = time.time()  # Record start time
max_time = 10  # Maximum time limit in seconds
person_detected = False  # Flag to track detection

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image. Exiting...")
        break  # Exit if the camera fails

    frame_count += 1  # Count frames

    # Preprocess frame
    image_resized = cv2.resize(frame, (224, 224))
    image_array = img_to_array(image_resized)
    image_expanded = np.expand_dims(image_array, axis=0)
    image_preprocessed = preprocess_input(image_expanded)

    # Perform object detection
    predictions = model.predict(image_preprocessed)
    decoded_predictions = decode_predictions(predictions, top=1)[0]

    # Display result on frame
    label = f"{decoded_predictions[0][1]}: {decoded_predictions[0][2] * 100:.2f}%"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("JDERobot Object Detection", frame)

    # ✅ Stop if 'q' is pressed (Manual Stop)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program...")
        break  

    # ✅ Stop if a person is detected with high confidence
    if decoded_predictions[0][1] == "person" and decoded_predictions[0][2] > 0.90:
        print("Person detected with high confidence. Stopping...")
        person_detected = True
        break  

    # ✅ Stop after a certain number of frames
    if frame_count >= max_frames:
        print(f"Max frame limit ({max_frames}) reached. Stopping...")
        break  

    # ✅ Stop if 10 seconds have passed
    if time.time() - start_time > max_time:
        print("Time limit (10 seconds) reached. Stopping...")
        break

# Show result if no detection occurred within time limit
if not person_detected:
    print("No required object detected within 10 seconds.")

# Release resources
cap.release()
cv2.destroyAllWindows()
