#not only say people are walking also describe the sorroundings which will help the visually impaired person


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
        temp_engine.setProperty('rate', 145) # কথা পরিষ্কার হওয়ার জন্য গতি সামান্য কমানো হয়েছে
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
    other_objects = []

    for r in results:
        # পোজ এবং অবজেক্ট ডিটেকশন থেকে তথ্য নেওয়া
        if r.boxes:
            frame = r.plot() # ভিজ্যুয়াল ফিডব্যাক
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                
                if label == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if (y2 - y1) > (x2 - x1) * 1.2:
                        walking_count += 1
                    else:
                        sitting_count += 1
                else:
                    # গাড়ি, গাছ বা অন্য গুরুত্বপূর্ণ অবজেক্টের নাম রাখা
                    if label in ['car', 'bus', 'truck', 'dog', 'bicycle', 'chair']:
                        other_objects.append(label)

    # বর্ণনা তৈরির লজিক
    current_time = time.time()
    if current_time - last_spoken_time >= 5: # দৃষ্টিহীন ব্যক্তির জন্য ৫ সেকেন্ড সময় ভালো যাতে তিনি প্রসেস করতে পারেন
        descriptions = []

        # মানুষের বর্ণনা
        if walking_count > 0:
            descriptions.append(f"{walking_count} people are walking nearby")
        if sitting_count > 0:
            descriptions.append(f"{sitting_count} person is sitting down")
        
        # অন্যান্য গুরুত্বপূর্ণ জিনিসের বর্ণনা
        if other_objects:
            unique_items = list(set(other_objects))
            descriptions.append(f"There is a {', '.join(unique_items)} in front of you")

        if descriptions:
            final_speech = " . ".join(descriptions)
            print(f"Describing: {final_speech}")
            
            t = threading.Thread(target=speak, args=(final_speech,))
            t.daemon = True
            t.start()
            
            last_spoken_time = current_time 

    cv2.imshow("Third Eye - Visual Assistant", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()