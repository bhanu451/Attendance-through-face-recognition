import cv2
import numpy as np
import imutils
import os
import pickle
import urllib.request

# Download required model files if missing
if not os.path.exists('deploy.prototxt'):
    print("Downloading deploy.prototxt...")
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
        'deploy.prototxt'
    )

if not os.path.exists('res10_300x300_ssd_iter_140000.caffemodel'):
    print("Downloading model weights...")
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel',
        'res10_300x300_ssd_iter_140000.caffemodel'
    )

# Load the face detection model
prototxt = 'deploy.prototxt'
model = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Directory to store user data
USER_DATA_DIR = 'user_data'
os.makedirs(USER_DATA_DIR, exist_ok=True)

# File to store trained faces
TRAINED_DATA_FILE = os.path.join(USER_DATA_DIR, 'trained_faces.pkl')

# Function to capture an image from webcam
def capture_photo():
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    print("Press 'SPACE' to capture a photo or 'ESC' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        cv2.imshow("Press SPACE to capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            cap.release()
            cv2.destroyAllWindows()
            return None
        elif key == 32:  # SPACE key to capture
            filename = os.path.join(USER_DATA_DIR, "captured.jpg")
            cv2.imwrite(filename, frame)
            cap.release()
            cv2.destroyAllWindows()
            return filename

# Detect faces in an image
def detect_faces(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    boxes = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            if face.shape[0] > 0 and face.shape[1] > 0:
                face = cv2.resize(face, (100, 100))
                faces.append(face)
                boxes.append((startX, startY, endX, endY))

    return faces, boxes

# Train a new face
def train_new_face():
    name = input("Enter your name: ")
    roll_no = input("Enter your roll number: ")
    print("Look at the camera and press 'SPACE' to capture your photo...")

    image_file = capture_photo()
    if not image_file:
        print("No photo captured.")
        return False

    image = cv2.imread(image_file)
    image = imutils.resize(image, width=400)

    faces, _ = detect_faces(image)
    if not faces:
        print("No clear face detected. Please try again.")
        return False

    face = faces[0]  # Use the first detected face

    try:
        with open(TRAINED_DATA_FILE, 'rb') as f:
            trained_faces = pickle.load(f)
    except FileNotFoundError:
        trained_faces = {}

    trained_faces[f"{name}_{roll_no}"] = face
    with open(TRAINED_DATA_FILE, 'wb') as f:
        pickle.dump(trained_faces, f)

    print(f"Face trained successfully for {name} (Roll No: {roll_no})!")
    return True

# Recognize faces
def recognize_faces():
    try:
        with open(TRAINED_DATA_FILE, 'rb') as f:
            trained_faces = pickle.load(f)
    except FileNotFoundError:
        print("No trained faces found. Please train first.")
        return

    print("Show your face to the camera for recognition...")

    image_file = capture_photo()
    if not image_file:
        print("No photo captured.")
        return

    image = cv2.imread(image_file)
    image = imutils.resize(image, width=400)

    detected_faces, boxes = detect_faces(image)
    recognized_names = set()

    for i, (face, box) in enumerate(zip(detected_faces, boxes)):
        startX, startY, endX, endY = box

        min_mse = float('inf')
        recognized_name = None

        for name_roll, trained_face in trained_faces.items():
            mse = np.mean((trained_face - face) ** 2)
            if mse < min_mse:
                min_mse = mse
                recognized_name = name_roll

        if recognized_name and min_mse < 1000:  # Recognition threshold
            text = f"{recognized_name}"
            color = (0, 255, 0)  # Green for recognized
            recognized_names.add(recognized_name)
        else:
            text = "Unknown"
            color = (0, 0, 255)  # Red for unrecognized

        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    cv2.imshow("Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if recognized_names:
        print("\nRecognized people:")
        for name in recognized_names:
            print(f"- {name}")

        confirm = input("Post attendance for these people? (y/n): ").lower()
        if confirm == 'y':
            print("Attendance posted for:")
            for name in recognized_names:
                print(f"- {name}")
        else:
            print("Attendance not posted.")
    else:
        print("No recognized faces.")

# Main program
print("Model files ready!")
while True:
    print("\n1. Train new face")
    print("2. Recognize faces and post attendance")
    print("3. Exit")
    choice = input("Enter your choice (1-3): ")

    if choice == '1':
        train_new_face()
    elif choice == '2':
        recognize_faces()
    elif choice == '3':
        break
    else:
        print("Invalid choice. Please try again.")
