import cv2
import cvzone
import math
import time
import numpy as np
import sys
import os
from ultralytics import YOLO

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

# Tracker Settings
# Maximum Distance: Very high (400px) to catch people even if FPS is very low and they jump across screen
TRACKER_DISTANCE_THRESHOLD = 400 

# Define Detection Lines
# These are horizontal lines on the floor where people cross.
# Moved closer (gap 30px) to ensure people don't skip the gap detection.
line1 = [(50, 235), (590, 235)] # Top Line (Entry Start)
line2 = [(50, 265), (590, 265)] # Bottom Line (Exit Start)

# ============================================
# 3. CAMERA SETUP (Picamera2)
# ============================================
from picamera2 import Picamera2

try:
    print("[INFO] Initializing Pi Camera (Picamera2)...")
    picam2 = Picamera2()
    # Configure camera to match our target resolution
    # CHANGED: Use BGR888 so OpenCV can display it directly without conversion (Fixes Color Issue)
    config = picam2.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"}
    )
    picam2.configure(config)
    picam2.start()
    print("[SUCCESS] Pi Camera started.")
    
except Exception as e:
    print("\n[FATAL ERROR] Failed to start Picamera2.")
    print(f"Error: {e}")
    print("Troubleshooting: Ensure 'picamera2' is installed and you are on a compatible Pi OS.")
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

tracker = Tracker()

# Counters
counter1 = [] # Entered
counter2 = [] # Exited
er = {} # Entry tracking
ex = {} # Exit tracking

# FPS Calculation
fps_start_time = time.time()
fps_counter = 0
current_fps = 0

# Debounce for Spam Prevention (Double Count Fix)
last_in_time = 0
last_out_time = 0
DEBOUNCE_DELAY = 1.0 # Seconds to wait before counting another person

# Mouse Callback for setup
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        print(f"POINT: [{x}, {y}]")

cv2.namedWindow('Pi People Counter', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Pi People Counter', RGB)

print("========================================")
print("  PI PEOPLE COUNTER RUNNING")
print("  Press 'ESC' to exit")
print("========================================")

# ============================================
# 5. MAIN LOOP
# ============================================
# LOGIC CONFIGURATION
LIMIT_LINE_Y = 240  # The horizontal line splitting Top/Bottom
SWAP_DIRECTIONS = False # Toggle this to fix "Opposite" counting

# State tracking: id -> "TOP" or "BOTTOM"
object_state = {} 

try:
    while True:
        # A. Read Frame (Picamera2)
        # Frame is already BGR (set in config), so no conversion needed.
        frame = picam2.capture_array()
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # REMOVED

        # C. YOLO Detection
        # Removed imgsz=320 because OpenVINO model is fixed to 640x640 (Static Shape)
        # Using default size to prevent shape mismatch error
        results = model(frame, stream=True, verbose=False, iou=0.4, conf=0.20, agnostic_nms=True)
        
        list_detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                
                # Confidence is already filtered by model(..., conf=0.25)
                # But we keep the loop structure
                cls = int(box.cls[0])
                currentClass = class_names[cls]

                # Filter for 'person'
                if (currentClass == 'person' or cls == 0):
                    list_detections.append([x1, y1, w, h])
        
        # D. Update Tracker
        # 'bbox_idx' contains [x, y, w, h, id] for all active people
        bbox_idx = tracker.update(list_detections)
        
        # E. Counting Logic (Hysteresis Buffer for Door)
        # CENTER OF DOOR (Blind Spot) = Dead Zone
        # Reduced Buffer back to 30 to catch people starting near the line
        BUFFER_Y = 30 # Total Dead Zone = 60px
        UP_LIMIT = 240 - BUFFER_Y   # 210
        DOWN_LIMIT = 240 + BUFFER_Y # 270
        for bbox in bbox_idx:
            x1, y1, w, h, id = bbox
            cx = int(x1 + x1 + w) // 2
            cy = int(y1 + y1 + h) // 2
            
            # Visuals: Clean Box + Explicit ID Label
            # Added Corner Rect to make tracking obvious
            cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=1, colorR=(255, 0, 255))
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cvzone.putTextRect(frame, f'ID: {id}', (x1, y1-35), 1, 1)

            # Determine Zone
            # Improved Logic: 
            # 1. If new object, assign state based on simple center line (so we don't start as 'UNKNOWN')
            # 2. If existing object, require them to cross the BUFFER limits to change state.
            
            if id not in object_state:
                # Initial Assignment (Startup fix)
                if cy < LIMIT_LINE_Y:
                    object_state[id] = "TOP"
                else:
                    object_state[id] = "BOTTOM"
            else:
                prev_state = object_state[id]
                current_state = prev_state # Default: hold state
                
                # Rigid Buffer Limits for Switching
                if cy < UP_LIMIT:
                    current_state = "TOP"
                elif cy > DOWN_LIMIT:
                    current_state = "BOTTOM"
                
                # Check Transition
                if current_state != prev_state:
                    # TOP -> BOTTOM
                    if prev_state == "TOP" and current_state == "BOTTOM":
                        if SWAP_DIRECTIONS:
                             if counter2.count(id) == 0:
                                 if (time.time() - last_out_time) > DEBOUNCE_DELAY:
                                     counter2.append(id)
                                     last_out_time = time.time()
                                     print(f"[EXIT] Person ID: {id} went OUT") # Console Output
                                     cv2.line(frame, (0, LIMIT_LINE_Y), (640, LIMIT_LINE_Y), (0, 0, 255), 5) 
                        else:
                             if counter1.count(id) == 0:
                                 if (time.time() - last_in_time) > DEBOUNCE_DELAY:
                                     counter1.append(id)
                                     last_in_time = time.time()
                                     print(f"[ENTRY] Person ID: {id} went IN") # Console Output
                                     cv2.line(frame, (0, LIMIT_LINE_Y), (640, LIMIT_LINE_Y), (0, 255, 0), 5) 

                    # BOTTOM -> TOP
                    elif prev_state == "BOTTOM" and current_state == "TOP":
                        if SWAP_DIRECTIONS:
                             if counter1.count(id) == 0:
                                 if (time.time() - last_in_time) > DEBOUNCE_DELAY:
                                     counter1.append(id)
                                     last_in_time = time.time()
                                     print(f"[ENTRY] Person ID: {id} went IN") # Console Output
                                     cv2.line(frame, (0, LIMIT_LINE_Y), (640, LIMIT_LINE_Y), (0, 255, 0), 5) 
                        else:
                             if counter2.count(id) == 0:
                                 if (time.time() - last_out_time) > DEBOUNCE_DELAY:
                                     counter2.append(id)
                                     last_out_time = time.time()
                                     print(f"[EXIT] Person ID: {id} went OUT") # Console Output
                                     cv2.line(frame, (0, LIMIT_LINE_Y), (640, LIMIT_LINE_Y), (0, 0, 255), 5) 

                    # Update State
                    object_state[id] = current_state

        # F. Draw Interface
        # Draw the Limit Line
        cv2.line(frame, (50, LIMIT_LINE_Y), (590, LIMIT_LINE_Y), (255, 255, 255), 2)
        cv2.putText(frame, "LINE", (50, LIMIT_LINE_Y-10), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
        
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
