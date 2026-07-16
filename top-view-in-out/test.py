import cv2
import cvzone
import math
import time
import numpy as np
import sys
import os
from ultralytics import YOLO
import requests
import json
import urllib3
import threading
from requests.auth import HTTPBasicAuth
from datetime import datetime
from flask import Flask, render_template, Response, jsonify

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================
# FLASK WEB SERVER CONFIGURATION
# ============================================
app = Flask(__name__)
outputFrame = None
lock = threading.Lock()

def generate():
    """Video streaming generator function."""
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                # Critical: Sleep to prevent 100% CPU usage loop
                time.sleep(0.1)
                continue
            
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                time.sleep(0.01)
                continue
                
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')
        # Limiter to prevent flooding
        time.sleep(0.03)

@app.route("/")
def index():
    return render_template("report.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/data")
def data():
    global counter1, counter2
    return jsonify({
        "in": len(counter1),
        "out": len(counter2)
    })

def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

# Start Flask in a background thread
t = threading.Thread(target=run_flask)
t.daemon = True
t.start()

# Automatically open the browser for the user
import webbrowser
def open_browser():
    time.sleep(2) # Wait for server to start
    webbrowser.open("http://127.0.0.1:5000")
    print("[INFO] Dashboard opened in browser.")

threading.Thread(target=open_browser, daemon=True).start()


# ============================================
# 1. TRACKER CLASS (Robust Version)
# ============================================
class Tracker:
    # Huge patience (40 frames) to keep ID even if detection flickers
    def __init__(self, max_disappeared=40):
        self.center_points = {} # id -> (cx, cy)
        self.disappeared = {}   # id -> frames_missing
        self.id_count = 0
        self.max_disappeared = max_disappeared

    def update(self, objects_rect):
        # objects_rect = [[x,y,w,h], ...]
        objects_bbs_ids = []

        # If no objects detected
        if len(objects_rect) == 0:
            for id in list(self.disappeared.keys()):
                self.disappeared[id] += 1
                if self.disappeared[id] > self.max_disappeared:
                    self.deregister(id)
            return objects_bbs_ids

        # Parse new input centroids
        input_centroids = []
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            input_centroids.append((cx, cy))

        # If we have no existing IDs, register all new inputs
        if len(self.center_points) == 0:
            for i in range(len(objects_rect)):
                self.register(i, input_centroids[i], objects_rect[i], objects_bbs_ids)
        
        else:
            object_ids = list(self.center_points.keys())
            object_centroids = list(self.center_points.values())
            
            # Simple Greedy Match
            # For each new input, find closest existing ID
            # This is O(N*M) but N,M are small (people count)
            
            # Keep track of used rows/cols
            used_existing_ids = set()
            
            # For each new observation
            for i, (in_cx, in_cy) in enumerate(input_centroids):
                min_dist = 99999
                best_id = -1
                
                # Check against all existing IDs
                for exist_id in object_ids:
                    if exist_id in used_existing_ids:
                        continue
                        
                    ex_cx, ex_cy = self.center_points[exist_id]
                    dist = math.hypot(in_cx - ex_cx, in_cy - ex_cy)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_id = exist_id
                
                # If we found a match within threshold
                if best_id != -1 and min_dist < TRACKER_DISTANCE_THRESHOLD:
                    self.center_points[best_id] = (in_cx, in_cy)
                    self.disappeared[best_id] = 0
                    objects_bbs_ids.append([objects_rect[i][0], objects_rect[i][1], objects_rect[i][2], objects_rect[i][3], best_id])
                    used_existing_ids.add(best_id)
                else:
                    # New ID
                    self.register(-1, (in_cx, in_cy), objects_rect[i], objects_bbs_ids)

            # Mark missing IDs as disappeared
            for exist_id in object_ids:
                if exist_id not in used_existing_ids:
                    self.disappeared[exist_id] += 1
                    if self.disappeared[exist_id] > self.max_disappeared:
                        self.deregister(exist_id)

        return objects_bbs_ids

    def register(self, index, centroid, rect, output_list):
        # Register new ID
        # If index is -1, generate new ID. otherwise unused.
        # Actually in this logic, we just use self.id_count
        
        self.center_points[self.id_count] = centroid
        self.disappeared[self.id_count] = 0
        output_list.append([rect[0], rect[1], rect[2], rect[3], self.id_count])
        self.id_count += 1

    def deregister(self, id):
        del self.center_points[id]
        del self.disappeared[id]

# ============================================
# 2. CONFIGURATION
# ============================================
# Using OpenVINO model for faster inference on CPU/Pi
MODEL_PATH = "inout_openvino_model/" 
# Lower confidence slightly for high-angle views (top-down people look different)
CONFIDENCE_THRESHOLD = 0.25 
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 15

# Maximum Distance: 120px (Optimal for preventing ID swapping)
TRACKER_DISTANCE_THRESHOLD = 120  # Was 400 in old version, too high causing ID swaps 

# Define Detection Lines
# These are horizontal lines on the floor where people cross.
# Moved closer (gap 30px) to ensure people don't skip the gap detection.
line1 = [(50, 235), (590, 235)] # Top Line (Entry Start)
line2 = [(50, 265), (590, 265)] # Bottom Line (Exit Start)

# ============================================
# 3. CAMERA SETUP (Hybrid: PiCamera2 or Webcam)
# ============================================
# Try importing Picamera2 (Works on Pi). If fail, use OpenCV (Windows/Laptop)
USE_PI_CAMERA = False
try:
    from picamera2 import Picamera2
    USE_PI_CAMERA = True
    print("[INFO] Picamera2 Library found. Attempting to initialize Pi Camera...")
except ImportError:
    print("[INFO] Picamera2 not found. Switching to Standard Webcam (Windows/Testing Mode)...")

try:
    if USE_PI_CAMERA:
        picam2 = Picamera2()
        # Configure camera to match our target resolution
        # CHANGED: Use BGR888 so OpenCV can display it directly without conversion
        config = picam2.create_preview_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        print("[SUCCESS] Pi Camera started via Picamera2.")
    else:
        # Fallback to standard webcam
        cap = cv2.VideoCapture(0)
        cap.set(3, FRAME_WIDTH) # Width
        cap.set(4, FRAME_HEIGHT) # Height
        if not cap.isOpened():
            raise Exception("Could not open Webcam.")
        print("[SUCCESS] Standard Webcam started.")

except Exception as e:
    print(f"\n[FATAL ERROR] Failed to start Camera.")
    print(f"Error: {e}")
    sys.exit(1)

# ============================================
# 4. INITIALIZE MODEL & TRACKER
# ============================================
try:
    print(f"[INFO] Loading YOLO model: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    class_names = model.names
    print("[SUCCESS] Model loaded.")
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    sys.exit(1)

# ZERO-MISS: 100 frame patience (very sticky IDs)
tracker = Tracker(max_disappeared=100)

# Counters
counter1 = [] # Entered
counter2 = [] # Exited
# Track each person's state for accurate counting
person_states = {}  # id -> {'direction': None, 'crossed_line': False, 'last_side': None}
# Track previous positions to determine direction of movement
previous_positions = {}  # id -> previous y-coordinate

# FPS Calculation
fps_start_time = time.time()
fps_counter = 0
current_fps = 0

# Balanced response for rapid entry/exit
last_in_time = 0
last_out_time = 0
DEBOUNCE_DELAY = 1.5

# ============================================
# OPENREMOTE CONFIGURATION & HELPER
# ============================================
OR_URL = "https://109.176.197.144/api/master/asset/55R1RiMqmqqNaFrVEoM8Ap/attribute/data"
OR_AUTH = HTTPBasicAuth("master:new_user", "p6nwLTG02ZRfjbMiNwDYzgGZd1G7OVmh")

def send_to_openremote(person_id, status):
    """
    Sends counting data to OpenRemote (Threaded).
    status: 'IN' or 'OUT'
    """
    def _send_thread():
        print(f"[DEBUG] Thread-Send ID {person_id} ({status})...")
        payload = {
            "id": str(person_id),
            "type": "PERSON",
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Short timeout to not hang the thread too long
            headers = {'Content-Type': 'application/json'}
            response = requests.put(OR_URL, json=payload, auth=OR_AUTH, headers=headers, verify=False, timeout=5)
            
            if response.status_code in [200, 204]:
                print(f"[OPENREMOTE] Success: ID {person_id} -> {status}")
            else:
                print(f"[OPENREMOTE] Error {response.status_code}: {response.text}")

        except Exception as e:
            print(f"[OPENREMOTE] Thread Failed: {e}")

    # Launch in background
    threading.Thread(target=_send_thread, daemon=True).start()

# Mouse Callback for setup
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        print(f"POINT: [{x}, {y}]")

cv2.namedWindow('Pi People Counter', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Pi People Counter', RGB)

print("========================================")
print("  PI PEOPLE COUNTER RUNNING")
print("  Mode: Anti-Ghost (9 FPS, High Confidence, Stable)")
print("  Press 'ESC' to exit")
print("  Dashboard: http://localhost:5000")
print("========================================")

# ============================================
# 5. MAIN LOOP
# ============================================
# LOGIC CONFIGURATION (Anti-Ghost Settings)
LIMIT_LINE_Y = 240 # Center point from snippet
SWAP_DIRECTIONS = False 
BUFFER_Y = 30 # Dead zone from snippet
UP_LIMIT = LIMIT_LINE_Y - BUFFER_Y
DOWN_LIMIT = LIMIT_LINE_Y + BUFFER_Y
DEBOUNCE_DELAY = 1.5 

frame_counter = 0     # For Performance monitoring
bbox_idx = []         # Tracker cache

try:
    while True:
        
        # A. Read Frame (Hybrid Logic)
        if USE_PI_CAMERA:
            # Capture RGB and flip to BGR permanently
            # .copy() is CRITICAL here to prevent "Layout incompatible with cv::Mat" crash
            frame = picam2.capture_array()
            frame = frame[:, :, ::-1].copy() # Instant color fix + Memory fix
        else:
            # Standard Webcam
            success, frame = cap.read()
            if not success:
                print("[ERROR] Failed to read frame from webcam.")
                break
        
        # C. YOLO Detection (Process every frame for better tracking)
        frame_counter += 1
        results = model(frame, stream=True, verbose=False, iou=0.3, conf=0.25, agnostic_nms=True)
        
        list_detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                
                cls = int(box.cls[0])
                currentClass = class_names[cls]

                # Filter for 'person' only
                if (currentClass == 'person' or cls == 0):
                    list_detections.append([x1, y1, w, h])

        # Remove duplicate detections (Non-Maximum Suppression like approach)
        filtered_detections = []
        for i, (x1, y1, w, h) in enumerate(list_detections):
            is_duplicate = False
            cx1, cy1 = x1 + w//2, y1 + h//2
            
            for j, (x2, y2, w2, h2) in enumerate(filtered_detections):
                cx2, cy2 = x2 + w2//2, y2 + h2//2
                distance = math.hypot(cx1 - cx2, cy1 - cy2)
                
                # If boxes overlap significantly, keep the larger one (likely better detection)
                if distance < 25:  # Reduced distance threshold for stricter filtering
                    is_duplicate = True
                    # Replace if current box is larger
                    if w * h > w2 * h2:
                        filtered_detections[j] = [x1, y1, w, h]
                    break
            
            if not is_duplicate:
                filtered_detections.append([x1, y1, w, h])
        
        list_detections = filtered_detections
        
        # Update Tracker every frame
        bbox_idx = tracker.update(list_detections)
        
        # E. Counting Logic & Visual Overlay (Runs on every frame)
        # ---------------------------------------------------------------
        for bbox in bbox_idx:
            x1, y1, w, h, id = bbox
            cx = int(x1 + x1 + w) // 2
            cy = int(y1 + y1 + h) // 2
            
            # VISUAL DEBUGGING: Show State
            state_text = "UNK"
            if id in person_states:
                state_text = person_states[id].get('last_side', 'UNK')
            
            # Visuals
            cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=1, colorR=(255, 0, 255))
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cvzone.putTextRect(frame, f'ID: {id} | {state_text}', (x1, y1-35), 1, 1)
            
            # Initialize person state if not already done
            if id not in person_states:
                person_states[id] = {
                    'direction': None,
                    'crossed_line': False,
                    'last_side': 'TOP' if cy < LIMIT_LINE_Y else 'BOTTOM'
                }
                previous_positions[id] = cy
                continue  # Don't count on first appearance, just initialize
            
            # Get previous position to determine direction
            prev_y = previous_positions[id]
            current_side = 'TOP' if cy < LIMIT_LINE_Y else 'BOTTOM'
            prev_side = person_states[id]['last_side']
            
            # Check for line crossing
            crossed_line = False
            crossing_direction = None  # 'down' or 'up'
            
            if prev_side == 'TOP' and current_side == 'BOTTOM':
                # Person moved from top to bottom (crossed line downward)
                crossed_line = True
                crossing_direction = 'down'
            elif prev_side == 'BOTTOM' and current_side == 'TOP':
                # Person moved from bottom to top (crossed line upward)
                crossed_line = True
                crossing_direction = 'up'
            
            # Only count if person crossed the line and hasn't been counted in this direction recently
            if crossed_line:
                person_state = person_states[id]
                
                # Determine if this is an IN or OUT based on direction and SWAP_DIRECTIONS
                is_entry = False
                if SWAP_DIRECTIONS:
                    # Swapped: down movement = OUT, up movement = IN
                    is_entry = (crossing_direction == 'up')
                else:
                    # Normal: down movement = IN, up movement = OUT
                    is_entry = (crossing_direction == 'down')
                
                # Apply debounce and count
                if is_entry:
                    if id not in counter1:  # Only count if not already counted as entered
                        if (time.time() - last_in_time) > DEBOUNCE_DELAY:
                            counter1.append(id)
                            last_in_time = time.time()
                            print(f"[ENTRY] ID {id}")
                            send_to_openremote(id, "IN")
                else:
                    if id not in counter2:  # Only count if not already counted as exited
                        if (time.time() - last_out_time) > DEBOUNCE_DELAY:
                            counter2.append(id)
                            last_out_time = time.time()
                            print(f"[EXIT] ID {id}")
                            send_to_openremote(id, "OUT")
                
                # Update person state after counting
                person_state['crossed_line'] = True
                person_state['direction'] = crossing_direction
                person_state['last_side'] = current_side
            
            # Update previous position for next frame comparison
            previous_positions[id] = cy

        # F. Draw Interface
        # Draw the Limit Line
        cv2.line(frame, (50, LIMIT_LINE_Y), (590, LIMIT_LINE_Y), (255, 255, 255), 2)
        cv2.putText(frame, "IN AREA", (50, LIMIT_LINE_Y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "OUT AREA", (50, LIMIT_LINE_Y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # FPS Calculation
        fps_counter += 1
        if fps_counter >= 10:
            current_fps = fps_counter / (time.time() - fps_start_time)
            fps_start_time = time.time()
            fps_counter = 0

        # UI Overlay
        count_in = len(counter1)
        count_out = len(counter2)
        cvzone.putTextRect(frame, f'In: {count_in} Out: {count_out} FPS: {int(current_fps)}', (20, 40), 1, 2)
        
        # Update Global Frame for Web Streaming
        with lock:
            outputFrame = frame.copy()
        
        cv2.imshow('Pi People Counter', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break

except KeyboardInterrupt:
    print("Stopped by user.")
except Exception as e:
    print(f"Runtime error: {e}")

finally:
    if 'picam2' in globals() and picam2:
        picam2.stop()
    cv2.destroyAllWindows()