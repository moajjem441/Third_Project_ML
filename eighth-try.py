# not only say a person ,,ai count the person and will say their activity


import cv2
from ultralytics import YOLO
import pyttsx3
import time
import threading

# YOLO Pose model load
model = YOLO("yolov8n-pose.pt")

def speak(text):
    try:
        temp_engine = pyttsx3.init()
        temp_engine.setProperty('rate', 150)
        temp_engine.say(text)
        temp_engine.runAndWait()
        del temp_engine
    except:
        pass

cap = cv2.VideoCapture("test_video_2.mp4")
last_spoken_time = 0 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)
    
    walking_count = 0
    sitting_count = 0

    for r in results:
        if r.keypoints is not None:
            # কঙ্কাল এবং বক্স ড্র করা
            frame = r.plot() 
            
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1

                # পোজ অনুযায়ী গণনা
                if h > w * 1.2:
                    walking_count += 1
                else:
                    sitting_count += 1

    # ৩ সেকেন্ড পরপর আপডেট লজিক
    current_time = time.time()
    if current_time - last_spoken_time >= 5:
        total_people = walking_count + sitting_count
        
        if total_people > 0:
            status_list = []
            
            # গ্রামার অনুযায়ী বাক্য সাজানো
            if walking_count > 0:
                p_text = "person" if walking_count == 1 else "people"
                status_list.append(f"{walking_count} {p_text} walking")
            
            if sitting_count > 0:
                p_text = "person" if sitting_count == 1 else "people"
                status_list.append(f"{sitting_count} {p_text} sitting")
            
            speech_text = "I see " + " and ".join(status_list)
            print(f"Update: {speech_text}")
            
            # ব্যাকগ্রাউন্ডে কথা বলা
            t = threading.Thread(target=speak, args=(speech_text,))
            t.daemon = True
            t.start()
            
            last_spoken_time = current_time 

    cv2.imshow("Third Eye - Multi-Person Mode", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()