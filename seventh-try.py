# video won't stop while speaking and voice repeat problem solved ,,




import cv2
from ultralytics import YOLO
import pyttsx3
import time
import threading

# YOLO Pose model load
model = YOLO("yolov8n-pose.pt")

# ব্যাকগ্রাউন্ডে কথা বলার ফাংশন
def speak(text):
    try:
        # প্রতিবার নতুন ইঞ্জিন ইনিশিয়ালাইজ করা হয় যাতে কনফ্লিক্ট না হয়
        temp_engine = pyttsx3.init()
        temp_engine.setProperty('rate', 150)
        temp_engine.say(text)
        temp_engine.runAndWait()
        # ইঞ্জিন ডিলিট করা হচ্ছে মেমোরি ক্লিয়ার রাখার জন্য
        del temp_engine
    except:
        pass

cap = cv2.VideoCapture("test_video_2.mp4")

# টাইমার শুরু
last_spoken_time = 0 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)
    current_action = ""

    for r in results:
        if r.keypoints is not None:
            frame = r.plot() 
            
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1

                # পোজ অনুযায়ী অ্যাকশন নির্ধারণ
                if h > w * 1.2:
                    current_action = "walking"
                else:
                    current_action = "sitting"

    # ৩ সেকেন্ড পর পর রিপিট করার লজিক
    current_time = time.time()
    if current_time - last_spoken_time >= 5: # ৩ সেকেন্ডের ব্যবধান
        if current_action != "":
            speech_text = f"The person is {current_action}"
            print(f"Update: {speech_text}")
            
            # থ্রেডিং ব্যবহার করে কথা বলা
            t = threading.Thread(target=speak, args=(speech_text,))
            t.daemon = True # মেইন প্রোগ্রাম বন্ধ হলে থ্রেডও বন্ধ হবে
            t.start()
            
            # সময় আপডেট করা
            last_spoken_time = current_time 

    cv2.imshow("Third Eye - 3 Sec Repeat Mode", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()