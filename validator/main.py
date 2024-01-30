import cv2
import os
import glob
from tkinter import Tk, Button
folder_path = './../my_images'

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
        image = cv2.imread(image_path)
        image = self.draw_points(image, points)

        # Display the current image number and total number of images
        position_text = f"Image {self.current_index + 1} of {len(self.image_paths)}"
        cv2.putText(image, position_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Image Validator", image)

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
