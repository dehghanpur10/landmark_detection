import os
import time
import tkinter as tk

import cv2

from const import *
from manipulate_img import center_crop, scale_image

class UI:
    def __init__(self, cap):
        self.current_type = None
        self.selected_point = {}
        self.capture_video = False
        self.counter = 0
        self.cap = cap
        self.window = tk.Tk()
        self.window.title(WINDOW_TITLE)
        self.window.geometry(WINDOW_SIZE)
        self.window.resizable(False, False)  # Prevent the window from being resized
        self.init_buttons()
        self.video_label = self.init_video_label()
        
    def select_type(self, type):
        self.current_type = type
        
    def clear_selected_point(self):
        self.selected_point = {}
    
    def set_capture_video(self, capture_video):
        self.capture_video = capture_video
        print(f"capture_video: {self.capture_video}")
    
    def init_buttons(self):
        button1 = tk.Button(self.window, text="Capture Snapshot", command=lambda: self.capture_snapshot())
        button1.grid(row=0, column=0)
        
        button5 = tk.Button(self.window, text="Clear", command=lambda: self.clear_selected_point())
        button5.grid(row=0, column=1)
        
        button6 = tk.Button(self.window, text="take video", command=lambda: self.set_capture_video(True))
        button6.grid(row=0, column=2)
        
        button7 = tk.Button(self.window, text="stop video", command=lambda: self.set_capture_video(False))
        button7.grid(row=0, column=3)

        button2 = tk.Button(self.window, text="Type 1", command=lambda: self.select_type(TYPE_ONE))
        button2.grid(row=1, column=0)
        
        button3 = tk.Button(self.window, text="Type 2", command=lambda: self.select_type(TYPE_TWO))
        button3.grid(row=1, column=1)

        button4 = tk.Button(self.window, text="Type 3", command=lambda: self.select_type(TYPE_THREE))
        button4.grid(row=1, column=2)
        
        button5 = tk.Button(self.window, text="Type 4", command=lambda: self.select_type(TYPE_FOUR))
        button5.grid(row=1, column=3)
        
        button5 = tk.Button(self.window, text="Type 5", command=lambda: self.select_type(TYPE_FIVE))
        button5.grid(row=2, column=0)
        
       
        
    def init_video_label(self):
        # Create a label to display the webcam feed
        label = tk.Label(self.window)
        label.bind('<Button-1>', self.handle_click)
        label.configure(cursor="target")
        label.grid(row=4, column=0, columnspan=4)
        return label
        
    def handle_click(self, event):
        # if self.current_type == None:
        #     print("any type isn't selected")
        #     return
        x, y = event.x, event.y
        print(x, y)

        if self.current_type == TYPE_ONE:
            self.selected_point[TYPE_ONE] = {
                'color': (0,0,255),
                'x': x,
                'y': y
            }
        elif self.current_type == TYPE_TWO:
            self.selected_point[TYPE_TWO] = {
                'color': (0,255,0),
                'x': x,
                'y': y
            }
        elif self.current_type == TYPE_THREE:
            self.selected_point[TYPE_THREE] = {
                'color': (255,0,0),
                'x': x,
                'y': y
            }
        elif self.current_type == TYPE_FOUR:
            self.selected_point[TYPE_FOUR] = {
                'color': (255,255,0),
                'x': x,
                'y': y
            }
        elif self.current_type == TYPE_FIVE:
            self.selected_point[TYPE_FIVE] = {
                'color': (255,0,255),
                'x': x,
                'y': y
            }
        
    def draw_circle(self,  frame):  
        for key in self.selected_point:
            point = self.selected_point[key]
            cv2.circle(frame, (point['x'], point['y']), 10, point['color'], 2)
    
    def capture_snapshot(self):
        # if len(self.selected_point) == 0:
        #     print("any point isn't selected")
        #     return
            
        filename = ""
        for key in self.selected_point:
            point = self.selected_point[key]
            type = key
            x = point['x']
            y = point['y']
            
            filename += f'{type}-{x}-{y}_'
        filename += f'{time.time()}.jpg'
        
        ret, frame = self.cap.read()
        if ret:
            print(frame.shape)
            # Specify the path to the directory where you want to save the image
            dirpath = os.path.expanduser("./../my_images")
            # Create the directory if it doesn't already exist
            os.makedirs(dirpath, exist_ok=True)
            
            # Specify the path where you want to save the image
            filepath = os.path.join(dirpath, filename)
            # Save the captured image
            scale_img = scale_image(frame, factor=1.1)
            crop_img = center_crop(scale_img, (512,512))
            cv2.imwrite(filepath, crop_img)
            print("Image saved to desktop")
            
           
 