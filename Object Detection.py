from ultralytics import YOLO
import cv2
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("Yolo_Weight/yolov8l.pt")

while True:
    success, img = cap.read()
    result = model(img, stream=True)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int (x1), int (y1), int (x2), int (y2) 
            print(x1, y1, x2, x2)
            cv2.rectangle(img,(x1,y1), (x2,y2), (255,0,255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)