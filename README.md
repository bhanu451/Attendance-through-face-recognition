# Face Recognition Attendance System

A Python-based attendance system that uses **OpenCV** and a **DNN face detector** to detect faces from a webcam and mark attendance automatically.

## Overview

This project is designed to simplify attendance tracking using real-time face recognition.  
It detects faces through a webcam, compares them with previously trained face data, and allows attendance to be recorded for recognized users.

## Features

- Train new faces using webcam input
- Real-time face detection and recognition
- Attendance marking after successful recognition
- Local storage of trained face data
- Lightweight and easy to run on a local machine

## Tech Stack

- Python
- OpenCV
- NumPy
- Imutils
- Pickle

## Project Files

```bash
Attendance-through-face-recognition/
├── README.md
├── hello.py
├── deploy.prototxt
└── res10_300x300_ssd_iter_14000.caffemodel
```

## How It Works

1. Run the Python script.
2. Choose to train a new face or recognize an existing one.
3. Capture face data using the webcam.
4. Compare detected faces with stored training data.
5. Mark attendance for recognized users.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/bhanu451/Attendance-through-face-recognition.git
cd Attendance-through-face-recognition
```

### 2. Install dependencies

```bash
pip install opencv-python numpy imutils
```

### 3. Run the project

```bash
python hello.py
```

## Usage

- Select the training option to register a new face.
- Enter the required details when prompted.
- Use the recognition option to detect and identify faces.
- Mark attendance once a face is successfully recognized.

## Output

The system opens the webcam feed, detects faces in real time, and shows recognition results directly on screen.

## Future Improvements

- Store attendance in CSV or database format
- Improve recognition accuracy with embeddings
- Add a graphical user interface
- Export attendance reports

## Author

**Bhanu**
- GitHub: [bhanu451](https://github.com/bhanu451)

## Repository Tags

`python` `opencv` `face-recognition` `attendance-system` `computer-vision`
