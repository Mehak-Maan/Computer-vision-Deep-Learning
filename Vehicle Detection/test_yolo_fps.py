from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import time

model = YOLO("models/numplate.onnx")

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280, 720)})
picam2.configure(config)
picam2.start()

fps_list = []

while True:
    t0 = time.time()

    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    model.predict(frame, conf=0.25, imgsz=640)

    fps = 1 / (time.time() - t0)
    fps_list.append(fps)

    if len(fps_list) > 30:
        print("FPS:", sum(fps_list)/len(fps_list))
        fps_list = []

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
