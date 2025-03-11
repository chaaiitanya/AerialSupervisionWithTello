Aerial Supervision: Real-Time Object Detection & Activity Recognition with Tello Drone

🚀 Overview

Aerial Supervision is a real-time object detection and human activity recognition system using a Tello drone. This project leverages YOLOv8 for object detection and MobileNetV2 for recognizing human activities.

It includes two versions:
	1.	aerialsupervision.py – Uses the real Tello drone for live video analysis.
	2.	mock.py – A simulated version using a webcam instead of a drone, allowing development and testing without hardware.

📌 Features

✅ Tello Drone Integration (aerialsupervision.py) – Controls takeoff, hover, and landing.
✅ Mock Mode (mock.py) – Uses a webcam for testing the code without a drone.
✅ Object Detection (YOLOv8) – Identifies people, vehicles, and objects in real-time.
✅ Human Activity Recognition (MobileNetV2) – Classifies human actions like walking, running, etc.
✅ Live Video Feed – Displays real-time annotations with bounding boxes.
✅ Metadata Collection – Logs detected object counts for analysis.

🛠️ Tech Stack
	•	djitellopy – Tello drone control.
	•	OpenCV – Video processing and visualization.
	•	YOLOv8 (Ultralytics) – Real-time object detection.
	•	MobileNetV2 (TensorFlow) – Human activity classification.

📜 How It Works

1️⃣ Tello Drone Version (aerialsupervision.py)
	1.	The drone takes off and starts streaming video.
	2.	YOLOv8 detects objects in the frame.
	3.	If a person is detected, MobileNetV2 predicts their activity.
	4.	The results are displayed on the screen with bounding boxes.
	5.	The drone lands safely when the program exits.

2️⃣ Mock Version (mock.py)
	1.	Uses the laptop webcam instead of a drone camera.
	2.	Runs the same object detection and HAR pipeline for testing.
	3.	Press ‘Q’ to stop the video feed.

📌 Installation

1️⃣ Install Dependencies

pip install djitellopy ultralytics opencv-python tensorflow

2️⃣ Run the Tello Drone Version

Make sure your Tello drone is connected via Wi-Fi. Then run:

python aerialsupervision.py

Press ‘Q’ to stop the program and land the drone.

3️⃣ Run the Mock (Webcam) Version

For testing without a drone, run:

python mock.py

This will use your webcam to simulate the drone feed.

🖼️ Demo Output

Live feed with detections:
📌 Bounding boxes for objects 📌 Predicted activity for humans

🛠️ Future Improvements

🚀 Train a custom YOLO model for improved accuracy.
🎯 Enhance HAR accuracy with fine-tuning.
🖐️ Add gesture control for drone navigation.
🗣️ Enable voice-based commands for controlling the drone.

📌 Contributions & Issues: Feel free to contribute via pull requests or report issues!
📌 License: MIT
