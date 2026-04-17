# detection threshold updated


import cv2
from ultralytics import YOLO
import pyttsx3
import time

# YOLO model load
model = YOLO("yolov8n.pt")

# voice engine setup
engine = pyttsx3.init()
engine.setProperty('rate', 140) 
engine.setProperty('volume', 1.0)

# video source
cap = cv2.VideoCapture("test_video.mp4")

last_spoken_time = 0 
speech_text = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Confidence threshold 0.5 set kora hoyeche jate bhul detection kom hoy
    results = model(frame, conf=0.5)
    
    detected_objects = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0]) # confidence score
            
            detected_objects.append(label)

            # bounding box draw
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    current_time = time.time()
    
    if current_time - last_spoken_time > 5:
        if detected_objects:
            unique_objects = list(set(detected_objects))
            
            if len(unique_objects) > 1:
                obj_text = ", ".join(unique_objects[:-1]) + " and " + unique_objects[-1]
            else:
                obj_text = unique_objects[0]
                
            speech_text = f"I see {obj_text}"
            
            print(f"--- [VOICE]: {speech_text} ---")
            engine.say(speech_text)
            engine.runAndWait()
            
            last_spoken_time = current_time

    if speech_text:
        cv2.putText(frame, f"AI Voice: {speech_text}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Third Eye", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()