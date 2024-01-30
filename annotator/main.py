import time
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import os
import UI
from manipulate_img import center_crop, scale_image


cap = cv2.VideoCapture(0)


UI = UI.UI(cap)
window = UI.window
video_label = UI.video_label






        
def capture_snapshot_without_point():
    
    ret, frame = cap.read()
    if ret:
         # Specify the path to the directory where you want to save the image
        dirpath = os.path.expanduser("~/Desktop/new_images")
        # Create the directory if it doesn't already exist
        os.makedirs(dirpath, exist_ok=True)
        
        # Specify the path where you want to save the image
        filepath = os.path.join(dirpath, f"{time.time()}.jpg")
        # Save the captured image
        scale_img = scale_image(frame, factor=1.1)

        ccrop_img = center_crop(scale_img, (512,512))
        cv2.imwrite(filepath, ccrop_img)
        print("Image saved to desktop")






# Function to update the webcam feed
def update_frame():

    ret, frame = cap.read()
    if ret:
        scale_img = scale_image(frame, factor=1.1)
        crop_img = center_crop(scale_img, (512,512))
        # Draw a circle on the frame
        # Parameters: (image, center_coordinates, radius, color, thickness)
        UI.draw_circle(crop_img)
        # print(frame.shape)        
        UI.counter+=1
        if UI.capture_video and UI.counter % 30 == 0:
            capture_snapshot_without_point()
        # Convert the frame to a format Tkinter can use
        
        cv_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
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
