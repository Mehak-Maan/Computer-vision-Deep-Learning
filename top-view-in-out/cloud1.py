import cv2
import cvzone
import math
import time
import numpy as np
import sys
import os
import threading
import requests
import webbrowser
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request
from requests.auth import HTTPBasicAuth
from ultralytics import YOLO
from picamera2 import Picamera2
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================
# 1. TRACKER CLASS (User's Robust Version)
# ============================================
class Tracker:
    def __init__(self, max_disappeared=40, tracking_distance=400):
        self.center_points = {} # id -> (cx, cy)
        self.disappeared = {}   # id -> frames_missing
        self.id_count = 0
        self.max_disappeared = max_disappeared
        self.tracking_distance = tracking_distance
        self.on_deregister = None

    def update(self, objects_rect):
        objects_bbs_ids = []
        if len(objects_rect) == 0:
            for id in list(self.disappeared.keys()):
                self.disappeared[id] += 1
                if self.disappeared[id] > self.max_disappeared: self.deregister(id)
            return objects_bbs_ids

        input_centroids = []
        for rect in objects_rect:
            x, y, w, h = rect
            input_centroids.append(((x + x + w) // 2, (y + y + h) // 2))

        if len(self.center_points) == 0:
            for i in range(len(objects_rect)):
                self.register(input_centroids[i], objects_rect[i], objects_bbs_ids)
        else:
            object_ids = list(self.center_points.keys())
            used_existing_ids = set()
            
            for i, (in_cx, in_cy) in enumerate(input_centroids):
                min_dist, best_id = 99999, -1
                for exist_id in object_ids:
                    if exist_id in used_existing_ids: continue
                    ex_cx, ex_cy = self.center_points[eid] if (eid := exist_id) in self.center_points else (999,999)
                    dist = math.hypot(in_cx - ex_cx, in_cy - ex_cy)
                    if dist < min_dist:
                        min_dist, best_id = dist, exist_id
                
                if best_id != -1 and min_dist < self.tracking_distance:
                    self.center_points[best_id] = (in_cx, in_cy)
                    self.disappeared[best_id] = 0
                    objects_bbs_ids.append([*objects_rect[i], best_id])
                    used_existing_ids.add(best_id)
                else:
                    self.register(input_centroids[i], objects_rect[i], objects_bbs_ids)

            for exist_id in object_ids:
                if exist_id not in used_existing_ids:
                    self.disappeared[exist_id] += 1
                    if self.disappeared[exist_id] > self.max_disappeared: self.deregister(exist_id)

        return objects_bbs_ids

    def register(self, centroid, rect, output_list):
        # SHIELD: Prevent shadow IDs by checking if someone was just here
        in_cx, in_cy = centroid
        for eid in self.center_points:
            if self.disappeared[eid] < 20: # Use 20 frame history
                ex_cx, ex_cy = self.center_points[eid]
                if math.hypot(in_cx - ex_cx, in_cy - ex_cy) < 80: # Balanced shield
                    return
        
        self.center_points[self.id_count] = centroid
        self.disappeared[self.id_count] = 0
        output_list.append([*rect, self.id_count])
        self.id_count += 1

    def deregister(self, id):
        if id in self.center_points: del self.center_points[id]
        if id in self.disappeared: del self.disappeared[id]
        if self.on_deregister: self.on_deregister(id)

# ============================================
# 2. DASHBOARD & SYSTEM CONFIG
# ============================================
app = Flask("PiCounter")
outputFrame = None
lock = threading.Lock()

# --- New OpenRemote Config ---
OR_BASE_URL = "https://109.176.197.144"
OR_REALM = "aiprojects"
OR_CLIENT_ID = "aiprojects"
OR_SECRET = "nV3eMyOSoIHCgRtqII0qiueDvgOCM670"
OR_ASSET_ID = "7fOIJbv6YcuzzXF9vCHgJj"

or_token = None
or_token_expiry = 0

def get_or_token():
    global or_token, or_token_expiry
    if or_token and time.time() < or_token_expiry:
        return or_token
    
    url = f"{OR_BASE_URL}/auth/realms/{OR_REALM}/protocol/openid-connect/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": OR_CLIENT_ID,
        "client_secret": OR_SECRET
    }
    try:
        r = requests.post(url, data=payload, verify=False, timeout=10)
        if r.status_code == 200:
            data = r.json()
            or_token = data.get("access_token")
            or_token_expiry = time.time() + data.get("expires_in", 3600) - 60
            return or_token
    except Exception as e:
        print(f"Auth Error: {e}")
    return None

counter1, counter2 = [], [] 
history_events = []
current_fps = 0

def add_event(id, status):
    global history_events
    event = {"id": id, "status": status, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    history_events.insert(0, event)
    if len(history_events) > 50: history_events.pop()

def send_to_openremote(id=None, status=None):
    def _send():
        token = get_or_token()
        if not token: return

        data_dict = {
            "id": str(id) if id is not None else "N/A",
            "status": status if status is not None else "HEARTBEAT", 
            "total_in": len(counter1), "total_out": len(counter2),
            "net_inside": max(0, len(counter1) - len(counter2)),
            "fps": int(current_fps),
            "timestamp": datetime.now().isoformat()
        }
        
        url = f"{OR_BASE_URL}/api/{OR_REALM}/asset/attributes"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = [{"ref": {"id": OR_ASSET_ID, "name": "Data"}, "value": data_dict}]
        
        try:
            requests.put(url, json=payload, headers=headers, verify=False, timeout=5)
        except Exception as e:
            print(f"OpenRemote Sync Error: {e}")
            
    threading.Thread(target=_send, daemon=True).start()

# ============================================
# 3. WEB ROUTES
# ============================================
@app.route("/")
def index(): return render_template("report.html")

@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with lock:
                if outputFrame is None: 
                    time.sleep(0.1); continue
                _, encodedImage = cv2.imencode(".jpg", outputFrame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            time.sleep(0.06)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/data")
def data():
    in_val, out_val = len(counter1), len(counter2)
    return jsonify({"in": in_val, "out": out_val, "fps": int(current_fps), "net_inside": max(0, in_val - out_val), "history": history_events})

@app.route("/edit", methods=["POST"])
def edit_event():
    global counter1, counter2
    req = request.json
    eid, new_s = int(req.get("id")), req.get("status")
    for e in history_events:
        if e["id"] == eid:
            old_s = e["status"]
            if old_s != new_s:
                if old_s == "IN" and eid in counter1: counter1.remove(eid)
                elif old_s == "OUT" and eid in counter2: counter2.remove(eid)
                if new_s == "IN": counter1.append(eid)
                else: counter2.append(eid)
                e["status"] = new_s
            break
    return jsonify({"success": True})

@app.route("/delete", methods=["POST"])
def delete_event():
    global counter1, counter2, history_events
    eid = int(request.json.get("id"))
    if eid in counter1: counter1.remove(eid)
    if eid in counter2: counter2.remove(eid)
    history_events = [e for e in history_events if e["id"] != eid]
    return jsonify({"success": True})

# ============================================
# 4. MAIN DETECTION ENGINE (Zero-Miss Logic)
# ============================================
def run_app():
    global outputFrame, counter1, counter2, current_fps
    
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
        picam2.configure(config)
        picam2.start()
    except Exception as e:
        print(f"[FATAL] Camera Error: {e}", flush=True); return

    print("[INFO] Loading YOLO model...", flush=True)
    model = YOLO("inout_openvino_model/", task="detect")
    class_names = model.names
    
    # Logic Configuration (Refined for Accuracy)
    UP_LINE_Y = 190
    DOWN_LINE_Y = 290
    
    object_start_side = {} 
    object_counted = {} # id -> bool
    last_count_location = [] # (x, y, timestamp)
    id_age = {} # id -> frames_seen (Anti-Ghost)
    
    # GLOBAL COOLDOWNS (Anti-Double-Count)
    last_in_time = 0
    last_out_time = 0
    
    def cleanup_id(id):
        if id in object_start_side: del object_start_side[id]
        if id in object_counted: del object_counted[id]
        if id in id_age: del id_age[id]

    # Tracker: bridge up to 600px jumps at lower FPS
    tracker = Tracker(max_disappeared=100, tracking_distance=600)
    tracker.on_deregister = cleanup_id
    
    fps_start_time, fps_counter = time.time(), 0
    cv2.namedWindow('Pi People Counter', cv2.WINDOW_NORMAL)

    try:
        while True:
            # Capture and fix color
            frame_raw = picam2.capture_array()
            # Convert BGR to RGB for YOLO processing and Display (to match User preference on this file)
            frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
            
            # Confidence 0.25 to reduce false positives (objects)
            # classes=[0] forces model to ONLY return Persons.
            # Reduced IOU to merge overlapping boxes better
            results = model(frame, stream=True, verbose=False, imgsz=640, iou=0.40, conf=0.25, agnostic_nms=True, classes=[0])
            
            list_detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1st, y1st, x2st, y2st = box.xyxy[0].cpu().numpy().astype(int)
                    # No need to check cls, we forced classes=[0] but double check safe
                    list_detections.append([x1st, y1st, x2st-x1st, y2st-y1st])

            bbox_idx = tracker.update(list_detections)
            now = time.time()
            # COOLDOWN 1.5s
            last_count_location = [loc for loc in last_count_location if (now - loc[2]) < 1.5]

            for bbox in bbox_idx:
                x1, y1, w, h, id = bbox
                
                # UPDATE AGE (Anti-Ghost)
                id_age[id] = id_age.get(id, 0) + 1
                
                # TRACKING POINT: Bottom Center (Feet)
                cx, cy = (x1 + x1 + w) // 2, (y1 + h)
                
                # Visuals
                cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=1, colorR=(255, 0, 255))
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1) 
                cvzone.putTextRect(frame, f'ID: {id}', (x1, y1-35), 1, 1)

                # 1. ORIGIN INITIALIZATION
                # If above UP_LINE -> Origin TOP
                # If below DOWN_LINE -> Origin BOTTOM
                if id not in object_start_side:
                    if cy < UP_LINE_Y: 
                        object_start_side[id] = "TOP"
                    elif cy > DOWN_LINE_Y: 
                        object_start_side[id] = "BOTTOM"
                    continue 

                # 2. TRIGGER LOGIC (Must be consistently seen for 5+ frames)
                if id_age.get(id, 0) > 4:
                    # Only count if NOT already counted for this pass
                    if not object_counted.get(id, False):
                        origin = object_start_side[id]
                        
                        # IN LOGIC: Started TOP, Crossed DOWN
                        if origin == "TOP" and cy > DOWN_LINE_Y:
                            # GLOBAL COOLDOWN CHECK
                            if (now - last_in_time) > 0.8:
                                # Debounce Check (80px radius)
                                is_blocked = False
                                for (bx, by, bt) in last_count_location:
                                    if math.hypot(cx - bx, cy - by) < 80:
                                        is_blocked = True; break
                                
                                if not is_blocked:
                                    counter1.append(id)
                                    object_counted[id] = True
                                    last_count_location.append((cx, cy, now))
                                    last_in_time = now # Update Global Timer
                                    
                                    add_event(id, "IN"); send_to_openremote(id, "IN")
                                    print(f"[IN] ID {id} Counted | Total IN: {len(counter1)}", flush=True)
                                    
                                    # RESET ORIGIN: Now they are at BOTTOM side
                                    object_start_side[id] = "BOTTOM"
                                    object_counted[id] = False # Allow immediate re-entry if they turn back

                        # OUT LOGIC: Started BOTTOM, Crossed UP
                        elif origin == "BOTTOM" and cy < UP_LINE_Y:
                            # GLOBAL COOLDOWN CHECK
                            if (now - last_out_time) > 0.8:
                                # Debounce Check (80px radius)
                                is_blocked = False
                                for (bx, by, bt) in last_count_location:
                                    if math.hypot(cx - bx, cy - by) < 80:
                                        is_blocked = True; break
                                
                                if not is_blocked:
                                    counter2.append(id)
                                    object_counted[id] = True
                                    last_count_location.append((cx, cy, now))
                                    last_out_time = now # Update Global Timer
                                    
                                    add_event(id, "OUT"); send_to_openremote(id, "OUT")
                                    print(f"[OUT] ID {id} Counted | Total OUT: {len(counter2)}", flush=True)
                                    
                                    # RESET ORIGIN: Now they are at TOP side
                                    object_start_side[id] = "TOP"
                                    object_counted[id] = False # Allow immediate re-entry if they turn back
            # Draw 2 Lines
            cv2.line(frame, (50, UP_LINE_Y), (590, UP_LINE_Y), (0, 255, 255), 2)   # Top Line
            cv2.line(frame, (50, DOWN_LINE_Y), (590, DOWN_LINE_Y), (0, 255, 255), 2) # Bottom Line
            cv2.putText(frame, "UP", (50, UP_LINE_Y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
            cv2.putText(frame, "DOWN", (50, DOWN_LINE_Y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
            fps_counter += 1
            if fps_counter >= 10:
                current_fps = fps_counter / (time.time() - fps_start_time)
                fps_start_time, fps_counter = time.time(), 0
            
            cvzone.putTextRect(frame, f'IN: {len(counter1)} OUT: {len(counter2)} FPS: {int(current_fps)}', (20, 40), 1, 2)
            
            with lock: outputFrame = frame
            cv2.imshow('Pi People Counter', frame)
            if cv2.waitKey(1) & 0xFF == 27: break
    finally:
        if 'picam2' in locals(): picam2.stop()
        cv2.destroyAllWindows()
if __name__ == '__main__':
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5001, debug=False, threaded=True), daemon=True).start()
    threading.Thread(target=lambda: (time.sleep(4), webbrowser.open("http://127.0.0.1:5001")), daemon=True).start()
    run_app()
