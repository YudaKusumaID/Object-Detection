import cv2
import requests
from PIL import Image

from ultralytics import YOLO

m3u8_url = "https://transcode.baliprov.dev/get-m3u8.php?srv=internals&channel=16&m3u8=live.m3u8"
response = requests.get(m3u8_url)
m3u8_content = response.content.decode('utf-8')

video_url = None
lines = m3u8_content.split('\n')
for line in lines:
    if line.startswith('http'):
        video_url = line
        break

if video_url is None:
    print("URL M3U8 TIDAK DITEMUKAN!")
    exit()

video_capture = cv2.VideoCapture(video_url)
model = YOLO('Yolo_Weight/yolov8l.pt')

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    results = model(pil_image, show=True)
