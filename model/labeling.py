import cv2
import os
import numpy as np
from enum import Enum


class TYPES(Enum):
    TYPE1 = 'type1'
    TYPE2 = 'type2'
    TYPE3 = 'type3'
    TYPE4 = 'type4'
    TYPE5 = 'type5'
    
def load_images_from_folder(folder):
    images = []
    names = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Modify as per your file types
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
                names.append(filename)

    return images, names