<h1 align="center">Face Recognition Attendance System</h1>

<p align="center">
  A Python-based attendance system that uses <b>OpenCV</b> and a <b>DNN face detector</b> to detect faces from a webcam and mark attendance automatically.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv" />
  <img src="https://img.shields.io/badge/Project-Attendance%20System-black?style=for-the-badge" />
</p>

---

## Overview

This project helps automate attendance tracking using face recognition.  
It captures facial data through a webcam, compares it with stored training data, and records attendance for recognized users.

## Features

- Real-time face detection using webcam input
- Face registration and training support
- Recognition of previously trained users
- Automatic attendance marking
- Lightweight local setup with simple dependencies

## Tech Stack

- Python
- OpenCV
- NumPy
- Imutils
- Pickle

## Project Structure

```bash
Attendance-through-face-recognition/
├── README.md
├── hello.py
├── deploy.prototxt
└── res10_300x300_ssd_iter_14000.caffemodel
```

## Workflow

1. Run the Python script.
2. Choose whether to train a new face or recognize an existing one.
3. Capture the face using the webcam.
4. Compare the detected face with stored data.
5. Mark attendance when a match is found.

## Installation

### Clone the repository

```bash
git clone https://github.com/bhanu451/Attendance-through-face-recognition.git
cd Attendance-through-face-recognition
```

### Install dependencies

```bash
pip install opencv-python numpy imutils
```

### Run the project

```bash
python hello.py
```

## Usage

- Register a new user by selecting the training option
- Enter the required details when prompted
- Start recognition mode to identify registered users
- Mark attendance once the face is recognized successfully

## Output

The system opens the webcam feed, detects faces in real time, and displays the recognition result on screen.

## Future Improvements

- Save attendance in CSV or database format
- Improve recognition accuracy with face embeddings
- Add a GUI for easier interaction
- Export attendance reports automatically

## Author

**Bhanu**

- GitHub: [bhanu451](https://github.com/bhanu451)

## Topics

`python` `opencv` `face-recognition` `attendance-system` `computer-vision`
