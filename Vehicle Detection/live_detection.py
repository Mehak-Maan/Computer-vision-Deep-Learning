import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
import time
from datetime import datetime
import os
import base64
import re
import threading
import queue
import json
import requests
import urllib3

from picamera2 import Picamera2

# Suppress SSL Warnings (Since we use raw IP)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# --- CONFIGURATION ---
# ==========================================

# FILES
VEHICLE_MODEL_PATH = "vehicle.onnx"
PLATE_MODEL_PATH = "numplate.onnx"
HTML_TEMPLATE_PATH = "report.html"
OUTPUT_HTML_PATH = "parking_report_live.html"
MEMORY_FILE = "parking_memory.json" 

# OPENREMOTE CREDENTIALS (HTTP)
OR_HOST = "https://109.176.197.144"  
OR_REALM = "master"
OR_CLIENT_ID = "new_user"           
OR_SECRET = "p6nwLTG02ZRfjbMiNwDYzgGZd1G7OVmh"
ASSET_ID = "4vC1DFDuGDdd44gB6z9D5B"
ATTRIBUTE_NAME = "data"

# PERFORMANCE TUNING
SKIP_FRAMES = 2
INFERENCE_SIZE = 320
CONF_THRESHOLD_VEHICLE = 0.50
CONF_THRESHOLD_PLATE = 0.30
INSTANT_LOG_CONFIDENCE = 0.50
MIN_PLATE_LENGTH = 3

# PARKING LOGIC
MIN_PARKING_DURATION_SECONDS = 30  # Half Minute

# ==========================================
# --- HTML TEMPLATE ---
# ==========================================
DEFAULT_STYLED_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visibility Bots Parking Log</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; padding: 20px; color: #343a40; }
        .container { max-width: 1200px; margin: 0 auto; background-color: #fff; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        header { text-align: center; margin-bottom: 30px; border-bottom: 2px solid #f8f9fa; padding-bottom: 20px; }
        h1 { color: #2c3e50; margin: 0; font-size: 2em; }
        table { width: 100%; border-collapse: separate; border-spacing: 0; margin-top: 20px; }
        th { background-color: #2c3e50; color: white; padding: 15px; text-align: left; font-weight: 600; }
        th:first-child { border-top-left-radius: 10px; }
        th:last-child { border-top-right-radius: 10px; }
        td { padding: 15px; border-bottom: 1px solid #eee; vertical-align: middle; }
        .vehicle-type { font-weight: 700; color: #2c3e50; display: block; margin-bottom: 4px; }
        .plate-badge { background-color: #3498db; color: white; padding: 4px 8px; border-radius: 4px; font-family: monospace; font-weight: 700; font-size: 0.9em; letter-spacing: 1px;}
        .vehicle-thumb { width: 100px; height: 75px; object-fit: cover; border-radius: 8px; border: 1px solid #ddd; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        
        /* Status Badges */
        .status-parked { color: #d35400; font-weight: 700; background: #ffeaa7; padding: 6px 12px; border-radius: 20px; display: inline-block; font-size: 0.85em; }
        .status-complete { color: #27ae60; font-weight: 700; background: #d4edda; padding: 6px 12px; border-radius: 20px; display: inline-block; font-size: 0.85em; }
        .status-complete::before { content: '✓ '; font-weight: 800; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Visibility Bots Smart Parking Solution</h1>
            <div style="color: #7f8c8d; font-size: 0.9em; margin-top: 5px;">Live Session Log</div>
        </header>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr><th>Vehicle</th><th>Snapshot</th><th>Entry</th><th>Exit</th><th>Duration</th><th>Status</th></tr>
                </thead>
                <tbody>
"""

class HTMLReporter:
    def __init__(self, template_path, output_path):
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            with open(self.output_path, 'w', encoding='utf-8') as f:
                f.write(DEFAULT_STYLED_HTML)
                f.write("</tbody></table></div></div></body></html>")

    def encode_image_to_base64(self, image):
        if image is None or image.size == 0: return ""
        try:
            _, buffer = cv2.imencode('.jpg', image)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{b64_str}"
        except: return ""

    def process_images_for_report(self, frame, vehicle_box, plate_box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, vehicle_box)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        vehicle_crop = frame[y1:y2, x1:x2]
        px1, py1, px2, py2 = map(int, plate_box)
        px1, py1, px2, py2 = max(0, px1), max(0, py1), min(w, px2), min(h, py2)
        plate_crop = frame[py1:py2, px1:px2]
        return self.encode_image_to_base64(vehicle_crop), self.encode_image_to_base64(plate_crop)

    def add_entry(self, vehicle_type, plate_text, time_in, time_out, duration, v_b64, status):
        # STYLING LOGIC
        if status == "IN":
            status_html = '<span class="status-parked">Parked</span>'
        else:
            status_html = '<span class="status-complete">Completed</span>'
            
        plate_display = plate_text if plate_text else "Unknown"

        new_row = f"""
        <tr>
            <td>
                <span class="vehicle-type">{vehicle_type}</span>
                <span class="plate-badge">{plate_display}</span>
            </td>
            <td><img src="{v_b64}" class="vehicle-thumb" alt="Vehicle"></td>
            <td>{time_in}</td>
            <td>{time_out}</td>
            <td>{duration}</td>
            <td>{status_html}</td>
        </tr>"""
        try:
            with open(self.output_path, 'r+', encoding='utf-8') as f:
                content = f.read()
                insert_pos = content.find('</tbody>')
                if insert_pos != -1:
                    f.seek(0)
                    f.write(content[:insert_pos] + new_row + content[insert_pos:])
                    f.truncate()
        except: pass

def log_local_report(reporter, v_obj, frame, status):
    """ Logs IN or OUT to HTML """
    time_now = datetime.now()
    time_str = time_now.strftime("%H:%M:%S")
    
    final_img = v_obj.frame_when_best_detected if v_obj.frame_when_best_detected is not None else frame
    v_b64, p_b64 = reporter.process_images_for_report(final_img, v_obj.best_vehicle_box, v_obj.best_plate_box)
    
    if status == "IN":
        reporter.add_entry(v_obj.type, v_obj.best_plate_text, time_str, "---", "Active", v_b64, "IN")
    else:
        # Calculate duration roughly
        reporter.add_entry(v_obj.type, v_obj.best_plate_text, "---", time_str, "Aborted", v_b64, "OUT")
    
    print(f"📄 [HTML LOG] Written: {v_obj.best_plate_text} ({status})")

# ==========================================
# --- SESSION MEMORY & LOGIC ---
# ==========================================
class ParkingManager:
    def __init__(self, filepath):
        self.filepath = filepath
        self.parked_cars = {} 
        self.load_memory()

    def load_memory(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    self.parked_cars = json.load(f)
                print(f"🧠 [MEMORY] Loaded {len(self.parked_cars)} cars currently parked.")
            except: self.parked_cars = {}

    def save_memory(self):
        try:
            with open(self.filepath, 'w') as f: json.dump(self.parked_cars, f)
        except: pass

    def process_plate(self, plate, vehicle_type):
        current_time = time.time()
        # CHECK EXIT
        if plate in self.parked_cars:
            entry_time = self.parked_cars[plate]
            duration = current_time - entry_time
            if duration > MIN_PARKING_DURATION_SECONDS:
                del self.parked_cars[plate] 
                self.save_memory()
                return "OUT"
            else:
                print(f"⏳ {plate} seen again too soon ({int(duration)}s). Ignoring.")
                return None 
        # ENTRY
        else:
            self.parked_cars[plate] = current_time
            self.save_memory()
            return "IN"

# ==========================================
# --- HTTP & SETUP ---
# ==========================================

ocr_queue = queue.Queue(maxsize=5) 
parking_manager = ParkingManager(MEMORY_FILE) 
VEHICLE_CLASSES = {0: 'Motorcycle', 1: 'Car', 2: 'Rickshaw', 3: 'Van', 4: 'Bus', 5: 'Truck'}

print("Loading OCR Engine...")
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

def get_openremote_token():
    token_url = f"{OR_HOST}/auth/realms/{OR_REALM}/protocol/openid-connect/token"
    payload = {'grant_type': 'client_credentials', 'client_id': OR_CLIENT_ID, 'client_secret': OR_SECRET}
    try:
        response = requests.post(token_url, data=payload, timeout=5, verify=False)
        if response.status_code == 200: return response.json().get("access_token")
    except: return None
    return None

def send_to_cloud(vehicle_type, plate_text, status):
    token = get_openremote_token()
    if not token: return
    payload = {"plate": plate_text, "type": vehicle_type, "status": status, "timestamp": datetime.now().isoformat()}
    api_url = f"{OR_HOST}/api/{OR_REALM}/asset/{ASSET_ID}/attribute/{ATTRIBUTE_NAME}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        requests.put(api_url, json=payload, headers=headers, timeout=5, verify=False)
        print(f"☁️ [CLOUD] Sent Status: {status} for {plate_text}")
    except Exception as e:
        print(f"⚠️ [CLOUD FAIL] {e}")

# ==========================================
# --- CORE PIPELINE ---
# ==========================================

def perform_ocr(plate_crop):
    if plate_crop.size == 0: return "", 0.0
    if plate_crop.shape[1] < 150:
        plate_crop = cv2.resize(plate_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    try:
        result = reader.readtext(gray, detail=1)
        if not result: return "", 0.0
        result.sort(key=lambda x: x[2], reverse=True)
        raw, conf = result[0][1], result[0][2]
        clean = re.sub(r'[^A-Z0-9]', '', raw.upper())
        print(f"[OCR] {clean} ({conf:.2f})")
        return clean, conf
    except: return "", 0.0

class TrackedVehicle:
    def __init__(self, track_id, vehicle_type, box):
        self.id = track_id; self.type = vehicle_type
        self.time_in_dt = datetime.now()
        self.best_plate_text = ""; self.best_plate_conf = 0.0
        self.last_seen_frame_count = 0; self.best_vehicle_box = box
        self.best_plate_box = None; self.frame_when_best_detected = None
        self.processed_logic = False 

    def update(self, box):
        self.last_seen_frame_count = 0; self.best_vehicle_box = box

    def add_plate_info(self, text, conf, plate_box, current_frame_img):
        if conf > self.best_plate_conf:
            self.best_plate_conf = conf; self.best_plate_text = text.upper()
            self.best_plate_box = plate_box; self.frame_when_best_detected = current_frame_img.copy()

def refine_vehicle_class(detected_class, x1, y1, x2, y2, img_w, img_h):
    width = x2-x1; height = y2-y1; area = width*height; aspect_ratio = width/height 
    if (detected_class == 'Bus' or detected_class == 'Truck') and area < 30000: return 'Car' 
    if detected_class == 'Bus' and aspect_ratio < 1.2: return 'Car'
    return detected_class

def ocr_worker(reporter):
    while True:
        try:
            data = ocr_queue.get(timeout=1) 
            plate_crop, abs_p_box, frame_ref, v_obj = data
            text, conf = perform_ocr(plate_crop)
            if conf > 0.15:
                v_obj.add_plate_info(text, conf, abs_p_box, frame_ref)
                if conf > INSTANT_LOG_CONFIDENCE and not v_obj.processed_logic:
                    status = parking_manager.process_plate(text, v_obj.type)
                    if status:
                        send_to_cloud(v_obj.type, text, status)
                        log_local_report(reporter, v_obj, frame_ref, status)
                    v_obj.processed_logic = True
            ocr_queue.task_done()
        except queue.Empty: continue
        except Exception as e: print(f"[WORKER ERROR] {e}")

def run_pipeline():
    print("Starting PiCamera2...")
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
    picam2.start()
    
    print(f"Loading Models...")
    vehicle_model = YOLO(VEHICLE_MODEL_PATH, task='detect')
    plate_model = YOLO(PLATE_MODEL_PATH, task='detect')

    reporter = HTMLReporter(HTML_TEMPLATE_PATH, OUTPUT_HTML_PATH)
    tracked_vehicles = {}
    
    print(f"Checking HTTP Connection...")
    if get_openremote_token(): print("✅ HTTP Auth Success!")
    else: print("⚠️ HTTP Auth Failed.")

    t = threading.Thread(target=ocr_worker, args=(reporter,), daemon=True)
    t.start()
    
    frame_count = 0
    try:
        while True:
            frame = picam2.capture_array()
            frame_count += 1
            display_frame = frame.copy(); h, w = frame.shape[:2]

            if frame_count % SKIP_FRAMES != 0: continue

            results = vehicle_model.track(frame, conf=CONF_THRESHOLD_VEHICLE, imgsz=INFERENCE_SIZE, persist=True, verbose=False)
            current_frame_ids = []

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                    current_frame_ids.append(track_id)
                    raw_type = VEHICLE_CLASSES.get(cls_id, 'Unknown')
                    v_type = refine_vehicle_class(raw_type, box[0], box[1], box[2], box[3], w, h)

                    if track_id not in tracked_vehicles: tracked_vehicles[track_id] = TrackedVehicle(track_id, v_type, box)
                    else: tracked_vehicles[track_id].update(box)
                    v_obj = tracked_vehicles[track_id]

                    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
                    vehicle_crop = frame[y1:y2, x1:x2]

                    if vehicle_crop.size > 0:
                        plate_results = plate_model(vehicle_crop, conf=CONF_THRESHOLD_PLATE, imgsz=320, verbose=False)
                        for p_box in plate_results[0].boxes.xyxy.cpu().numpy().astype(int):
                            px1, py1, px2, py2 = p_box
                            abs_p_box = [x1+px1, y1+py1, x1+px2, y1+py2]
                            cv2.rectangle(display_frame, (abs_p_box[0], abs_p_box[1]), (abs_p_box[2], abs_p_box[3]), (0, 255, 0), 2)
                            if not ocr_queue.full() and v_obj.best_plate_conf < 0.95 and not v_obj.processed_logic:
                                plate_crop = vehicle_crop[py1:py2, px1:px2].copy()
                                ocr_queue.put( (plate_crop, abs_p_box, frame.copy(), v_obj) )

            ids_to_remove = []
            for tid, v_obj in tracked_vehicles.items():
                if tid not in current_frame_ids:
                    v_obj.last_seen_frame_count += 1
                    if v_obj.last_seen_frame_count > 30: ids_to_remove.append(tid)
            for tid in ids_to_remove: del tracked_vehicles[tid]

            cv2.imshow("Visibility Bots - Live Parking Stream", display_frame)
            if cv2.waitKey(1) == ord('q'): break

    except KeyboardInterrupt: pass
    finally:
        picam2.stop(); cv2.destroyAllWindows(); print("System Stopped.")

if __name__ == "__main__":
    run_pipeline()
