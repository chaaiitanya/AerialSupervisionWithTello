import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from ultralytics import YOLO


# Mock Tello class to simulate drone behavior
class MockTello:
    def connect(self):
        print("[Mock] Drone connected")

    def streamon(self):
        print("[Mock] Video streaming started")

    def takeoff(self):
        print("[Mock] Drone taking off...")

    def land(self):
        print("[Mock] Drone landing...")

    def streamoff(self):
        print("[Mock] Video streaming stopped")


drone = MockTello()  # Use MockTello instead of real drone

# Load Pre-trained Models
mobilenet_model = MobileNetV2(weights="imagenet")  # Use MobileNetV2 for HAR
yolo_model = YOLO("yolov8n.pt")  # YOLOv8 for object detection

# Metadata storage
metadata = {}


# Preprocess frames for HAR model
def preprocess_frame_for_har(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame_array = img_to_array(frame_rgb)
    frame_preprocessed = preprocess_input(frame_array)
    return np.expand_dims(frame_preprocessed, axis=0)


# Predict human activity using MobileNetV2
def predict_activity(frame):
    preprocessed = preprocess_frame_for_har(frame)
    prediction = mobilenet_model.predict(preprocessed)

    # Decode predictions (Top-5 class labels)
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
    decoded_predictions = decode_predictions(prediction, top=1)
    activity_label = decoded_predictions[0][0][1]  # Get the most likely label

    return activity_label


# Detect objects using YOLO
def detect_objects(frame):
    results = yolo_model(frame)
    detections = results[0].boxes.data.cpu().numpy()

    object_list = []
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        label = yolo_model.names[int(cls)]
        object_list.append(
            {"xmin": int(x1), "ymin": int(y1), "xmax": int(x2), "ymax": int(y2), "name": label, "confidence": conf})

    return object_list


# Update metadata
def update_metadata(detections):
    for obj in detections:
        label = obj["name"]
        metadata[label] = metadata.get(label, 0) + 1


# Webcam-based simulation
def simulate_with_webcam():
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("[Error] Could not access webcam.")
        return

    try:
        drone.takeoff()
        print("[Mock] Drone is hovering...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Error] Failed to capture frame")
                break

            detections = detect_objects(frame)

            # Check for human presence
            if any(obj["name"] == "person" for obj in detections):
                print("Human(s) detected!")
                activity = predict_activity(frame)
                print(f"Predicted Activity: {activity}")

            update_metadata(detections)

            # Draw bounding boxes
            for obj in detections:
                cv2.rectangle(frame, (obj["xmin"], obj["ymin"]), (obj["xmax"], obj["ymax"]), (0, 255, 0), 2)
                cv2.putText(frame, f"{obj['name']} ({obj['confidence']:.2f})", (obj["xmin"], obj["ymin"] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Webcam Simulation", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"[Error] {e}")

    finally:
        drone.land()
        cap.release()
        cv2.destroyAllWindows()
        print("[Mock] Drone landed safely.")
        print("Metadata collected:", metadata)


# Run the simulation
if __name__ == "__main__":
    simulate_with_webcam()