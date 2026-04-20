# object detection and activity describe only for human

import cv2
from ultralytics import YOLO
import pyttsx3
import time

# YOLO Pose model  (এটি মানুষের হাড়ের জয়েন্ট ট্র্যাক করে)
model = YOLO("yolov8n-pose.pt")

engine = pyttsx3.init()
engine.setProperty('rate', 150)

cap = cv2.VideoCapture("test_video.mp4")
last_spoken_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    # Pose Detection
    results = model(frame, conf=0.5)
    current_actions = []
    

    for r in results:
        # Keypoints (শরীরের জয়েন্টগুলো)
        if r.keypoints is not None:
            keypoints = r.keypoints.data.cpu().numpy()
            
            for person_keypoints in keypoints:
                # পায়ের গোড়ালির (Ankle) মুভমেন্ট বা পজিশন চেক করা যায়
                # তবে সহজ করার জন্য আমরা এখনো বক্স রেশিও এবং পোজ এর সমন্বয় করছি
                x1, y1, x2, y2 = map(int, r.boxes.xyxy[0])
                w = x2 - x1
                h = y2 - y1

                # পোজ অনুযায়ী উন্নত লজিক
                if h > w * 1.2:
                    action = "walking"
                else:
                    action = "sitting"
                
                current_actions.append(action)
                
                # কঙ্কালের মতো পোজ ড্র করবে (এটি দেখতে খুব কুল লাগে!)
                frame = r.plot() 

    # ৫ সেকেন্ড পরপর ভয়েস আপডেট
    current_time = time.time()
    if current_time - last_spoken_time > 5 and current_actions:
        unique_action = list(set(current_actions))[0]
        speech_text = f"The person is {unique_action}"
        
        print(speech_text)
        engine.say(speech_text)
        engine.runAndWait()
        last_spoken_time = current_time

    cv2.imshow("Third Eye - Pose Action", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()