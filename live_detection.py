# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import cv2
from keras.models import load_model
import pandas as pd
import os
from ultralytics import YOLO
from datetime import datetime
from scipy.spatial import distance

model = load_model("age_sex_detection_rmsprop.keras")
input_shape = (70, 70) 
yolo_face_model = YOLO("yolov8n-face-lindevs.pt")

top = tk.Tk()
top.geometry('800x600')
top.title('Age & Gender Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background="#CDCDCD", font=('arial', 15))
label2 = Label(top, background="#CDCDCD", font=('arial', 15))
sign_image = Label(top)

tracked_faces = {}  # dictionary to store face data
next_face_id = 0    # unique id counter

def get_face_id(new_face, existing_faces):
    """Finds a matching existing face or assigns a new ID."""
    global next_face_id
    (x1, y1, x2, y2) = new_face
    new_center = ((x1 + x2) // 2, (y1 + y2) // 2)

    for face_id, data in existing_faces.items():
        (ex1, ey1, ex2, ey2) = data["coords"]
        existing_center = ((ex1 + ex2) // 2, (ey1 + ey2) // 2)

        if distance.euclidean(new_center, existing_center) < 50:
            return face_id  

    # If no match, assign a new ID
    face_id = next_face_id
    next_face_id += 1
    return face_id

def Detect():
    """Detects faces in live video feed and predicts age & gender (only once per person)."""
    global label_packed
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        label1.configure(foreground="red", text="Error: Unable to access camera.")
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
                faces.append((x1, y1, x2, y2))

        sex_f = ["Male", "Female"]

        for face in faces:
            face_id = get_face_id(face, tracked_faces) 
            x1, y1, x2, y2 = face

       
            if face_id not in tracked_faces:
                face_crop = frame[y1:y2, x1:x2]
                face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, input_shape)
                face_resized = face_resized.astype('float32') / 255.0
                face_resized = np.expand_dims(face_resized, axis=-1)
                face_resized = np.expand_dims(face_resized, axis=0)

           
                pred = model.predict(face_resized)
                age = int(np.round(pred[1][0])) 
                sex = int(np.round(pred[0][0])) 

                tracked_faces[face_id] = {
                    "coords": face,
                    "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "exit_time": None,
                    "age": age,
                    "sex": sex_f[sex]
                }

            else:
                # update last seen location & exit time
                tracked_faces[face_id]["coords"] = face
                tracked_faces[face_id]["exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # retrieve stored predictions
            age = tracked_faces[face_id]["age"]
            sex = tracked_faces[face_id]["sex"]

            # Draw rectangle & labels
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {age}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Gender: {sex}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = frame.resize((400, 400))
        im = ImageTk.PhotoImage(frame)

        sign_image.configure(image=im)
        sign_image.image = im
        top.update()

    cap.release()


    df = pd.DataFrame.from_dict(tracked_faces, orient='index')
    df.to_csv('output.csv', index=False)


live_camera = Button(top, text="Start Live Detection", command=Detect, padx=10, pady=5)
live_camera.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
live_camera.pack(side='bottom', pady=50)

# UI Layout
sign_image.pack(side='bottom', expand=True)
label1.pack(side="bottom", expand=True)
label2.pack(side="bottom", expand=True)

heading = Label(top, text="Age and Gender Detector", pady=20, font=('arial', 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()


top.mainloop()
