import tkinter as tk

import cv2

from const import *

class UI:
    def __init__(self, cap):
        self.current_person = MOHAMMAD
        self.person_change = 0
        self.counter = 0
        self.cap = cap
        self.window = tk.Tk()
        self.window.title(WINDOW_TITLE)
        self.window.geometry(WINDOW_SIZE)
        self.window.resizable(False, False)  # Prevent the window from being resized
        self.video_label = self.init_video_label()
    def init_video_label(self):
        # Create a label to display the webcam feed
        label = tk.Label(self.window)
        label.pack()
        # label.grid(row=3, column=0, columnspan=4)
        return label
        
    def handle_person_possibility(self, person_possibility):
        if self.current_person == MOHAMMAD:
            if person_possibility > 0.3:
                self.person_change = 0
            else:
                self.person_change += 1
            
            if self.person_change == 5:
                self.current_person = HAMED
                self.person_change = 0
                    
        elif self.current_person == HAMED:
            if person_possibility < 0.7:
                self.person_change = 0
            else:
                self.person_change += 1
            
            if self.person_change == 5:
                self.current_person = MOHAMMAD
                self.person_change = 0
                
    def draw_circle(self,  frame,x,y,color ):  
        cv2.circle(frame, (x, y), 20, color, 2)