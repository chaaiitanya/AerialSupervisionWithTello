import cv2
import numpy as np
from djitellopy import Tello
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from ultralytics import YOLO

# -----------------------------
# 🚁 Initialize Tello Drone
# -----------------------------
drone = Tello()
drone.connect()
drone.streamon()

# -----------------------------
# 🧠 Load Pretrained Models
# -----------------------------
mobilenet_model = MobileNetV2(weights="imagenet")  # Pretrained MobileNetV2 for activity recognition
yolo_model = YOLO("yolov8n.pt")  # YOLOv8 for object detection

# Metadata storage for detected objects
metadata = {}

# -----------------------------
# 🔄 Preprocessing for HAR Model
# -----------------------------
def preprocess_frame_for_har(frame):
    """
    Prepares the frame for MobileNetV2 by resizing, normalizing, and converting color space.
    """
    frame_resized = cv2.resize(frame, (224, 224))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame_array = img_to_array(frame_rgb)
    frame_preprocessed = preprocess_input(frame_array)
    return np.expand_dims(frame_preprocessed, axis=0)

# -----------------------------
# 🏃 Predict Human Activity
# -----------------------------
def predict_activity(frame):
    """
    Uses MobileNetV2 to classify human activity based on the frame.
    """
    preprocessed = preprocess_frame_for_har(frame)
    prediction = mobilenet_model.predict(preprocessed)
    decoded_predictions = decode_predictions(prediction, top=1)  # Get top-1 prediction
    activity_label = decoded_predictions[0][0][1]  # Extract label name
    return activity_label

# -----------------------------
# 📦 Detect Objects Using YOLO
# -----------------------------
def detect_objects(frame):
    """
    Uses YOLOv8 to detect objects in the frame and return bounding boxes.
    """
    results = yolo_model(frame)
    detections = results[0].boxes.data.cpu().numpy()

    object_list = []
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        label = yolo_model.names[int(cls)]
        object_list.append({"xmin": int(x1), "ymin": int(y1), "xmax": int(x2), "ymax": int(y2), "name": label, "confidence": conf})

    return object_list

# -----------------------------
# 🗂️ Update Metadata for Objects
# -----------------------------
def update_metadata(detections):
    """
    Updates metadata dictionary with object counts detected in the scene.
    """
    for obj in detections:
        label = obj["name"]
        metadata[label] = metadata.get(label, 0) + 1

# -----------------------------
# ✈️ Drone Hover & Analysis
# -----------------------------
def hover_and_analyze():
    """
    The main function where the drone takes off, hovers, and performs real-time object detection & activity recognition.
    """
    try:
        # 🚁 Take off
        drone.takeoff()
        print("🚁 Drone has taken off and is hovering...")

        while True:
            # 🎥 Get live video feed from drone camera
            frame = drone.get_frame_read().frame

            # 🔍 Detect objects in the frame using YOLOv8
            detections = detect_objects(frame)

            # 🏃‍♂️ If humans are detected, predict their activity
            if any(obj["name"] == "person" for obj in detections):
                print("👀 Human(s) detected!")
                activity = predict_activity(frame)
                print(f"🏃 Predicted Activity: {activity}")

            # 📦 Update metadata with detected objects
            update_metadata(detections)

            # 🖼️ Draw bounding boxes and labels on the frame
            for obj in detections:
                cv2.rectangle(frame, (obj["xmin"], obj["ymin"]), (obj["xmax"], obj["ymax"]), (0, 255, 0), 2)  # Green box
                cv2.putText(frame, f"{obj['name']} ({obj['confidence']:.2f})", (obj["xmin"], obj["ymin"] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 🖥️ Display the live feed with annotations
            cv2.imshow("Tello Live Feed", frame)

            # 🔴 Stop execution if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"❌ Error occurred: {e}")

    finally:
        # 🚁 Land the drone safely
        drone.land()
        drone.streamoff()
        cv2.destroyAllWindows()
        print("✅ Drone landed safely.")
        print("📊 Metadata collected:", metadata)

# -----------------------------
# 🚀 Run the Main Function
# -----------------------------
if __name__ == "__main__":
    hover_and_analyze()