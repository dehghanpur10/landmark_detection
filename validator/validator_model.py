import cv2
import os
import glob
from tkinter import Tk, Button
from keras.models import load_model
from keras.utils import CustomObjectScope
import numpy as np
import tensorflow as tf


INPUT_SHAPE = 128
SCALE = INPUT_SHAPE / 512.0
folder_path = './../my_images'

def scale_image(img, factor=1):
	"""Returns resize image by scale factor.
	This helps to retain resolution ratio while resizing.
	Args:
	img: image to be scaled
	factor: scale factor to resize
    
	"""
	return cv2.resize(img,(int(img.shape[1]*factor), int(img.shape[0]*factor)))
def normalize_input_images(images):
    # Convert images to float32
    images = images.astype('float32')
    images = images / 127.5 - 1
    return images


with CustomObjectScope():
    model = load_model('../model/models/model.h5')
    
class ImageValidator:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_paths = glob.glob(os.path.join(folder_path, '*.jpg'))
        self.current_index = 0
        self.setup_ui()

    def setup_ui(self):
        self.window = Tk()
        self.window.title("Image Validator")
        Button(self.window, text="Previous", command=self.show_previous_image).pack(side='left')
        Button(self.window, text="Delete", command=self.delete_image).pack(side='left')
        Button(self.window, text="Next", command=self.show_next_image).pack(side='left')
        self.show_image()

    def parse_filename(self, filename):
        parts = os.path.basename(filename).split('_')
        points = {'type1': [], 'type2': [], 'type3': [], 'type4': [], 'type5': []}
        for part in parts[:-1]:  # Exclude the last part (timestamp)
            if 'type' in part:
                type_id, x, y = part.split('-')
                points[type_id].append((int(x), int(y)))
        print(f"Parsed points from {filename}: {points}")  # Debugging print
        return points

    def draw_points(self, image, points):
        print 
        colors = {'type1': (0, 0, 255), 'type2': (0, 255, 0), 'type3': (255, 0, 0), 'type4': (255, 255, 0), 'type5': (255, 0, 255)}
        for type_id, pts in points.items():
            for point in pts:
                cv2.circle(image, point, 5, colors[type_id], -1)
                print(f"Drew {type_id} at {point}")  # Debugging print
        return image

    def show_image(self):
        if self.current_index < 0 or self.current_index >= len(self.image_paths):
            print("No more images.")
            return
        image_path = self.image_paths[self.current_index]
        points = self.parse_filename(image_path)
        cv_img = cv2.imread(image_path)
        a = scale_image(cv_img, SCALE)
        image = normalize_input_images(a)
        image = np.expand_dims(image, axis=0)
        print(type(image))
        # load model and predict
        prediction = model.predict(image)
        
        output = prediction[0]
        x1, y1 = output[0], output[1]
        x2, y2 = output[2], output[3]
        x3, y3= output[4], output[5]
        x4, y4 = output[6], output[7]
        x5, y5 = output[8], output[9]
        # person_possibility = output[10]
        person = "MOHAMMAD"
        # if (person_possibility < 0.5):
        #     person = "HAMED"
        # else:
        #     person = "MOHAMMAD"
        
        (x, y, w, h) = (10, 0, 200, 40)  # Adjust the position and size of the rectangle as needed
        cv2.rectangle(cv_img, (x, y), (x + w, y + h), (255, 255, 255), -1)  # Create a white rectangle
        cv2.putText(cv_img, person, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        
        cv2.circle(cv_img, (int(x1*512), int(y1*512)), 15, (255,0,0), 2)
        cv2.circle(cv_img, (int(x2*512), int(y2*512)), 15, (0,255,0), 2)
        cv2.circle(cv_img, (int(x3*512), int(y3*512)), 15, (0,0,255), 2)
        cv2.circle(cv_img, (int(x4*512), int(y4*512)), 15, (255,255,0), 2)
        cv2.circle(cv_img, (int(x5*512), int(y5*512)), 15, (255,0,255), 2)

        # Display the current image number and total number of images
        position_text = f"Image {self.current_index + 1} of {len(self.image_paths)}"
        cv2.putText(cv_img, position_text, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Image Validator", cv_img)

    def show_next_image(self):
        self.current_index += 1
        self.show_image()

    def show_previous_image(self):
        self.current_index -= 1
        self.show_image()

    def delete_image(self):
        if 0 <= self.current_index < len(self.image_paths):
            os.remove(self.image_paths[self.current_index])
            del self.image_paths[self.current_index]
            self.show_next_image()

    def run(self):
        self.window.mainloop()

# Usage
app = ImageValidator(folder_path)
app.run()
