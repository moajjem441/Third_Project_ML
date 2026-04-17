import cv2
from ultralytics import YOLO
import pyttsx3


# YOLO model load (first time download hobe)
model = YOLO("yolov8n.pt")


# voice engine
engine = pyttsx3.init()


# video file (tomar video path dao)
cap = cv2.VideoCapture("test_video.mp4")

last_spoken = ""


while True:
    ret, frame = cap.read()
    if not ret:
        break


    # object detection
    results = model(frame)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # same kotha bar bar na bolar jonno
            if label != last_spoken:
                text = f"{label} ahead"
                print(text)

                engine.say(text)
                engine.runAndWait()

                last_spoken = label

            # bounding box draw
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)


    cv2.imshow("Third Eye", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()