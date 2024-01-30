import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import UI
from const import HAMED, MOHAMMAD
from manipulate_img import center_crop, scale_image
from keras.models import load_model
from keras.utils import CustomObjectScope

INPUT_SHAPE = 128
SCALE = INPUT_SHAPE / 512.0

def normalize_input_images(images):
    # Convert images to float32
    images = images.astype('float32')
    images = images / 127.5 - 1
    return images

cap = cv2.VideoCapture(0)


UI = UI.UI(cap)
window = UI.window
video_label = UI.video_label


with CustomObjectScope():
    model = load_model('../model/models/model.h5')
    model_b = load_model('../model/models/model_b.h5')

# Function to update the webcam feed
def update_frame():

    ret, frame = cap.read()
    if ret:
        scale_img = scale_image(frame, factor=1.1)
        crop_img = center_crop(scale_img, (512,512))        
        
        cv_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        image = normalize_input_images(scale_image(cv_img,SCALE))
        image = np.expand_dims(image, axis=0)
        
        prediction = model.predict(image)
        output = prediction[0]
        x1, y1 = output[0], output[1]
        x2, y2 = output[2], output[3]
        x3, y3= output[4], output[5]
        x4, y4 = output[6], output[7]
        x5, y5 = output[8], output[9]
        
        prediction_b = model_b.predict(image)
        output = prediction_b[0]

        person_possibility = output[0]
        print(person_possibility)
        UI.handle_person_possibility(person_possibility)
        (x, y, w, h) = (10, 0, 200, 40)  # Adjust the position and size of the rectangle as needed
        cv2.rectangle(cv_img, (x, y), (x + w, y + h), (255, 255, 255), -1)  # Create a white rectangle
        cv2.putText(cv_img, UI.current_person, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        
        cv2.circle(cv_img, (int(x1*512), int(y1*512)), 25, (255,0,0), 2)
        cv2.circle(cv_img, (int(x2*512), int(y2*512)), 25, (0,255,0), 2)
        cv2.circle(cv_img, (int(x3*512), int(y3*512)), 15, (0,0,255), 2)
        cv2.circle(cv_img, (int(x4*512), int(y4*512)), 15, (255,255,0), 2)
        cv2.circle(cv_img, (int(x5*512), int(y5*512)), 15, (255,0,255), 2)

        img = Image.fromarray(cv_img , mode="RGB")

        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    video_label.after(1, update_frame)



    
    


# Start updating the webcam feed
update_frame()


# Start the GUI
window.mainloop()

# Release the webcam when closing the app
cap.release()
cv2.destroyAllWindows()
