🧠 Face Recognition Attendance System
📘 Overview

This project is a Face Recognition Attendance System built using OpenCV and Deep Neural Networks (DNN) for facial detection.
It allows users to train their faces and mark attendance automatically by recognizing faces from a webcam feed.

The system uses OpenCV’s SSD-based deep learning face detector (res10_300x300_ssd_iter_140000.caffemodel) and compares new faces with trained ones using Mean Squared Error (MSE) similarity.

🚀 Features

🧍‍♂️ Train new faces using your webcam

🎥 Real-time face detection and recognition

💾 Stores trained faces in a local .pkl file

🗂️ Automatically downloads the required OpenCV model files if missing

✅ Option to mark attendance after successful recognition

🧰 Lightweight and easy to run locally



🧩 Project Structure

face_recognition_attendance/
│
├── user_data/  

│   └── trained_faces.pkl           

│
├── deploy.prototxt    

├── res10_300x300_ssd_iter_140000.caffemodel

├── face_recognition_attendance.py 

├── README.md    

└── requirements.txt                 


⚙️ Installation & Setup

1️⃣ Clone the Repository

hello.py

2️⃣ Install Required Libraries

Create a virtual environment (recommended) and install dependencies:

pip install -r requirements.txt


requirements.txt

opencv-python
imutils
numpy

3️⃣ Run the Program
python face_recognition_attendance.py

🧭 Usage Instructions

Once you run the script, you’ll see the following menu:

1. Train new face
2. Recognize faces and post attendance
3. Exit

🧍 Train a New Face

Enter your name and roll number.

Press SPACE to capture your face using the webcam.

The system will detect and store your face in user_data/trained_faces.pkl.

🎥 Recognize Faces

Show your face to the webcam.

The system will detect faces and compare them with the trained data.

If recognized, it displays your name and asks whether to mark attendance.

🧠 Technical Details

Face Detection: OpenCV DNN (SSD-based model)

Face Comparison Metric: Mean Squared Error (MSE)

Data Storage: Pickle serialization (trained_faces.pkl)

Threshold: MSE < 1000 for recognition

Dependencies: OpenCV, NumPy, Imutils, Pickle

📁 Data Files Auto-Downloaded

If missing, the script automatically downloads:

deploy.prototxt

res10_300x300_ssd_iter_140000.caffemodel
