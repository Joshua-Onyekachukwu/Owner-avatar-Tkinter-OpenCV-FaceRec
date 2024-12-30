import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Create a directory to store face images
dataset_dir = 'face_dataset'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Function to capture images of the face and save them with unique labels
def capture_face_images(name, num_images=50):
    cap = cv2.VideoCapture(0)
    count = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y + h, x:x + w]
            file_name = f"{dataset_dir}/{name}_{count}.jpg"
            cv2.imwrite(file_name, face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing {name} - {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the frame with face bounding box and text
        cv2.imshow("Face Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= num_images:
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Info", f"Captured {count} images of {name}")


# Function to train the face recognizer
def train_face_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels, names = [], [], {}
    label_counter = 0

    # Load images and labels from dataset
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".jpg"):
            img_path = os.path.join(dataset_dir, filename)
            gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces.append(gray_img)

            name = filename.split('_')[0]  # Assuming filenames are in the format "Name_number.jpg"
            if name not in names:
                names[name] = label_counter
                label_counter += 1
            labels.append(names[name])

    recognizer.train(faces, np.array(labels))
    recognizer.save('face_recognizer.yml')
    messagebox.showinfo("Info", "Model trained and saved as 'face_recognizer.yml'")

    # Return names dictionary for later use
    return names


# Function to recognize faces in real-time using the trained model
def recognize_face(names):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('face_recognizer.yml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            label, confidence = recognizer.predict(face_img)

            if confidence < 50:  # Confidence threshold for good recognition
                name = list(names.keys())[label]  # Get name from label
                color = (0, 255, 0)  # Green for recognized face
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red for unrecognized face

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# GUI application using Tkinter
def start_gui():
    # Create the main window
    root = tk.Tk()
    root.title("Face Recognition System")

    def on_capture():
        name = name_entry.get()
        if name:
            capture_face_images(name)
        else:
            messagebox.showerror("Error", "Please enter a name.")

    def on_train():
        names = train_face_recognizer()

    def on_recognize():
        names = train_face_recognizer()  # Train first before recognizing
        recognize_face(names)

    # Create GUI components
    name_label = tk.Label(root, text="Enter your name:")
    name_label.pack(padx=10, pady=5)

    name_entry = tk.Entry(root)
    name_entry.pack(padx=10, pady=5)

    capture_button = tk.Button(root, text="Capture Face", command=on_capture)
    capture_button.pack(padx=10, pady=5)

    train_button = tk.Button(root, text="Train Model", command=on_train)
    train_button.pack(padx=10, pady=5)

    recognize_button = tk.Button(root, text="Start Recognition", command=on_recognize)
    recognize_button.pack(padx=10, pady=5)

    # Run the Tkinter event loop
    root.mainloop()


# Start the GUI
start_gui()
