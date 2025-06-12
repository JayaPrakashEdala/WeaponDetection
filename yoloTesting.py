from ultralytics import YOLO
import cv2
import cvzone
import math
from playsound import playsound
import time
import pygame
import threading
model = YOLO(r"D:\code\aiexperiments\finalProject\trainedVersions\400epoch\weights\best.pt")

pygame.mixer.init()

video = cv2.VideoCapture(0)

def play_alert_sound():
    # Play a simple alert sound using pygame
    pygame.mixer.music.load('alarm.mp3')  # Replace 'alert_sound.wav' with your sound file
    pygame.mixer.music.play()
    time.sleep(5)  # Adjust duration as needed
    pygame.mixer.music.stop()

def async_play_alert_sound():
    # Run the play_alert_sound function in a separate thread
    alert_thread = threading.Thread(target=play_alert_sound)
    alert_thread.start()
    # time.sleep(5)

def detect():
    isSoundPlaying = False
    weapon = False

    while True:
        data, frame = video.read()
        res = model(frame, stream=True)
        for r in res:
            boxes = r.boxes
            for box in boxes:
              #bounding box
              x1,y1,x2,y2 = box.xyxy[0]
              x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
              w,h = x2-x1,y2-y1
              cvzone.cornerRect(frame,(x1,y1,w,h))
            #   confidence
              conf = math.ceil((box.conf[0]*100))/100
              #className
              cls = int(box.cls[0])
              cvzone.putTextRect(frame,f'{model.names[cls]} {conf}',(max(0,x1),max(35,y1)),scale=0.7,thickness=1)
              if conf >= 0.7:
                  weapon = True
                  async_play_alert_sound()            
        cv2.imshow("image",frame)
        cv2.waitKey(1)

detect()