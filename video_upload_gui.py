# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import cv2
from keras.models import load_model
import pandas as pd
from ultralytics import YOLO
from datetime import datetime

# Load models
model = load_model("C:/Users/Admin/Desktop/nullclass/nullclass_gender_age_detection/age_sex_detection_rmsprop.keras")
input_shape = (70, 70)
yolo_face_model = YOLO("C:/Users/Admin/Desktop/nullclass/nullclass_gender_age_detection/yolov8n-face-lindevs.pt")

# Initialize Tkinter
top = tk.Tk()
top.geometry('800x600')
top.title('Age & Gender Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background="#CDCDCD", font=('arial', 15))
label2 = Label(top, background="#CDCDCD", font=('arial', 15))
sign_image = Label(top)

tracked_faces = {}  # dictionary to store face data
next_face_id = 0    # unique id counter

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def get_face_id(new_face, existing_faces):
    global next_face_id
    (x, y, w, h) = new_face

    for face_id, data in existing_faces.items():
        (ex, ey, ew, eh) = data["coords"]
        iou = calculate_iou((x, y, w, h), (ex, ey, ew, eh))

        if iou > 0.5:  # Adjust threshold as needed
            return face_id  

    # If no match, assign a new ID
    face_id = next_face_id
    next_face_id += 1
    return face_id

def Detect(video_path):
    global label_packed
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        label1.configure(foreground="red", text="Error: Unable to open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_face_model(frame)
        faces = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                faces.append((x1, y1, x2 - x1, y2 - y1))

        sex_f = ["Male", "Female"]
        for (x, y, w, h) in faces:
            face_id = get_face_id((x, y, w, h), tracked_faces)

            if face_id not in tracked_faces:
                face = frame[y:y + h, x:x + w]
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, input_shape)
                face_resized = face_resized.astype('float32') / 255.0
                face_resized = np.expand_dims(face_resized, axis=-1)
                face_resized = np.expand_dims(face_resized, axis=0)

                pred = model.predict(face_resized)
                age = int(np.round(pred[1][0]))
                sex = int(np.round(pred[0][0]))

                if age > 55:
                    tracked_faces[face_id] = {
                        "coords": (x, y, w, h),
                        "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "exit_time": None,
                        "age": age,
                        "sex": sex_f[sex]
                    }
            else:
                
                if face_id in tracked_faces:
                    tracked_faces[face_id]["coords"] = (x, y, w, h)
                    tracked_faces[face_id]["exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if face_id in tracked_faces:
                age = tracked_faces[face_id]["age"]
                sex = tracked_faces[face_id]["sex"]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Age: {age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Gender: {sex}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = frame.resize((400, 400))
        im = ImageTk.PhotoImage(frame)

        sign_image.configure(image=im)
        sign_image.image = im
        top.update()

    cap.release()

    filtered_faces = {face_id: data for face_id, data in tracked_faces.items() if data["age"] > 55}

    df = pd.DataFrame.from_dict(filtered_faces, orient='index')
    df.to_csv('C:/Users/Admin/Desktop/nullclass/nullclass_gender_age_detection/output.csv', index=False)

def show_Detect_button(file_path):
    Detect_b = Button(top, text="Detect Video", command=lambda: Detect(file_path), padx=10, pady=5)
    Detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    Detect_b.place(relx=0.79, rely=0.46)

def upload_video():
    try:
        file_path = filedialog.askopenfilename()
        show_Detect_button(file_path)
    except Exception as e:
        label1.configure(foreground="red", text=f"Error: {e}")

# UI Layout
upload = Button(top, text="Upload a Video", command=upload_video, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=50)

sign_image.pack(side='bottom', expand=True)
label1.pack(side="bottom", expand=True)
label2.pack(side="bottom", expand=True)

heading = Label(top, text="Age and Gender Detector", pady=20, font=('arial', 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

top.mainloop()