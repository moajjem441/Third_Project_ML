# voice repeat but not happen

import cv2
from ultralytics import YOLO
import pyttsx3
import time


# YOLO Pose model  load
model = YOLO("yolov8n-pose.pt")

# Voice engine setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

cap = cv2.VideoCapture("test_video.mp4")

# save the record of last talks
last_spoken_time = time.time() 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)
    current_action = ""

    for r in results:
        if r.keypoints is not None:
            # পোজ অনুযায়ী ফ্রেম প্লট করা (কঙ্কাল দেখাবে)
            frame = r.plot() 
            
            # বক্সের মাপ নিয়ে অ্যাকশন চেক করা
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1

                if h > w * 1.2:
                    current_action = "walking"
                else:
                    current_action = "sitting"

    # ৫ সেকেন্ড পরপর কড়া আপডেট লজিক
    current_time = time.time()
    
    # যদি ৫ সেকেন্ড পার হয় এবং কোনো অ্যাকশন খুঁজে পাওয়া যায়
    if current_time - last_spoken_time >= 2:
        if current_action != "":
            speech_text = f"The person is {current_action}"
            print(f"Update: {speech_text}")
            
            engine.say(speech_text)
            engine.runAndWait()
            
            # কথা বলার পর বর্তমান সময়কে আবার 'লাস্ট টাইম' হিসেবে সেট করা
            last_spoken_time = time.time() 

    cv2.imshow("Third Eye - Constant Voice Update", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()