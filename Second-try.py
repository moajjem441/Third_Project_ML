# voice control and updated


import cv2
from ultralytics import YOLO
import pyttsx3
import time


# YOLO model load
model = YOLO("yolov8n.pt")


# voice engine setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # গলার স্বর পরিষ্কার করার জন্য স্পিড কমানো হয়েছে (for clearing the voice or vocal cord)

# video source
cap = cv2.VideoCapture("test_video.mp4")

last_spoken_time = 0 
last_spoken_label = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # বর্তমান সময় চেক করা
            current_time = time.time()

            # যদি নতুন অবজেক্ট হয় অথবা একই অবজেক্ট ৩ সেকেন্ড পর আবার বলতে চায়
            if label != last_spoken_label or (current_time - last_spoken_time > 3):
                text = f"I see a {label}"
                print(text)
                
                engine.say(text)
                engine.runAndWait()
                
                last_spoken_label = label
                last_spoken_time = current_time

            # Draw
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Third Eye", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # 'q' চেপে বন্ধ করা যাবে
        break

cap.release()
cv2.destroyAllWindows()