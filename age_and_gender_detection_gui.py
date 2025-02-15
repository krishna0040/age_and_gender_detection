# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import cv2
from keras.models import load_model
import pandas as pd 
import os
from ultralytics import YOLO  


# Load your trained age & gender model
model = load_model("model.keras")                
#model input is 70 x 70 
input_shape = (70,70)
# Load YOLOv8 face detection model (pre-trained model for face detection)
yolo_face_model = YOLO("C:/Users/Admin/Desktop/nullclass/nullclass_gender_age_detection/yolov8n-face-lindevs.pt")  

top = tk.Tk()
top.geometry('800x600')
top.title('Age & Gender Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background="#CDCDCD", font=('arial', 15))
label2 = Label(top, background="#CDCDCD", font=('arial', 15))
sign_image = Label(top)

data_old_age = []

def Detect(file_path):
    global label_packed
    try:
        image = cv2.imread(file_path)
        if image is None:
            raise Exception("Unable to load image. Please check the file path.")

        # Use YOLO to detect faces
        results = yolo_face_model(image)
        
        faces = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                faces.append((x1, y1, x2 - x1, y2 - y1))  

        if len(faces) == 0:
            raise Exception("No faces detected in the image.")

        sex_f = ["Male", "Female"]
        for (x, y, w, h) in faces:
            
            face = image[y:y + h, x:x + w]
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)           
            face_resized = cv2.resize(face_gray, input_shape)

            face_resized = face_resized.astype('float32') / 255.0
            face_resized = np.expand_dims(face_resized, axis=-1)  
            face_resized = np.expand_dims(face_resized, axis=0)   

           
            pred = model.predict(face_resized)
            age = int(np.round(pred[1][0]))  
            sex = int(np.round(pred[0][0]))  

            if age >= 55:  
                data_old_age.append([age, sex, faces])

            # Draw rectangle and labels on the original image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"Age: {age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Gender: {sex_f[sex]}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # Convert image for Tkinter display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((400, 400))  # Resize for display
        im = ImageTk.PhotoImage(image)

        # Update GUI
        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(foreground="#011638", text=f"Age: {age}")
        label2.configure(foreground="#011638", text=f"Gender: {sex_f[sex]}")

    except Exception as e:
        label1.configure(foreground="red", text=f"Error: {e}")
        label2.configure(text="")

    # Save detected data to CSV
    columns_labels = ['age', 'sex', 'frame_data']
    df = pd.DataFrame(data_old_age, columns=columns_labels)
    df.to_csv('C:/Users/Admin/Desktop/nullclass/nullclass_gender_age_detection/output.csv', index=False)

# Function to create a Detect button after image upload
def show_Detect_button(file_path):
    Detect_b = Button(top, text="Detect Image", command=lambda: Detect(file_path), padx=10, pady=5)
    Detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    Detect_b.place(relx=0.79, rely=0.46)

# Function to upload an image
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        label2.configure(text='')
        show_Detect_button(file_path)
    except Exception as e:
        label1.configure(foreground="red", text=f"Error: {e}")
        label2.configure(text="")

# Upload button
upload = Button(top, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=50)

# Packing the image and labels
sign_image.pack(side='bottom', expand=True)
label1.pack(side="bottom", expand=True)
label2.pack(side="bottom", expand=True)

# Heading Label
heading = Label(top, text="Age and Gender Detector", pady=20, font=('arial', 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

# Start the Tkinter GUI
top.mainloop()
