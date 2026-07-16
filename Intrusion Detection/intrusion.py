import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import sys
import os
import json
import time
import threading
import datetime
import webbrowser
import queue
import shutil
from flask import Flask, render_template, Response, jsonify, request
import requests
import base64
import collections
import paho.mqtt.client as mqtt
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ===================== FLASK SETUP ====================
app = Flask(__name__, template_folder=".")
shutdown_flag = False
reset_flag = False

def get_local_ip():
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

LOCAL_IP = get_local_ip()

# ===================== GLOBALS ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "zone_config.json")
STATS_FILE = os.path.join(BASE_DIR, "intrusion_stats.json")
COORDS_LOG_FILE = os.path.join(BASE_DIR, "intrusion_coords.txt")
CAPTURE_DIR = os.path.join(BASE_DIR, "static/captures")
STATIC_DIR = os.path.join(BASE_DIR, "static")

stats = {
    "total_intrusions": 0,
    "last_breach": "--:--:--",
    "alerts_sent": 0,
    "logs": [],
    "zone_history": [],
    "current_snapshot_url": "",
    "current_snapshot_b64": "",
    "fps": 0.0,
    "uptime": "00:00:00",
    "system_start": time.time(),
    "zone_active": True
}

# Shared State for Web Feed and Zone
outputFrame = None
lock = threading.Lock()
task_queue = queue.Queue()

# Shared Zone Configuration
zone_pts = [[100,100], [540,100], [540,400], [100,400]]
current_zone = Polygon(zone_pts)
zone_lock = threading.Lock()

# ===================== OPENREMOTE MQTT CONFIG ============
MQTT_BROKER = "109.176.197.144"
MQTT_PORT = 1883
MQTT_USER = "aiprojects:new_user"
MQTT_PASS = "p6nwLTG02ZRfjbMiNwDYzgGZd1G7OVmh"
MQTT_CLIENT_ID = "client-id"
MQTT_TOPIC = "aiprojects/client-id/writeattributevalue/Data/6RY5HlP2bSmn3UXwM7PAuX"

mqtt_client = None

def init_mqtt():
    global mqtt_client
    try:
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, MQTT_CLIENT_ID)
        mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print(">>> MQTT Client Initialized and Connected. <<<")
        return True
    except Exception as e:
        print(f"FAILED TO INIT MQTT: {e}")
        return False

# ===================== OPENREMOTE REST CONFIG ============
OR_BASE_URL = "https://109.176.197.144"
OR_REALM = "aiprojects"
OR_CLIENT_ID = "aiprojects"
OR_SECRET = "nV3eMyOSoIHCgRtqII0qiueDvgOCM670"
OR_ASSET_ID = "6RY5HlP2bSmn3UXwM7PAuX"

# Store for token
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

def sync_to_openremote(data_dict):
    # 1. Sync via REST (Verified Method)
    token = get_or_token()
    if token:
        url = f"{OR_BASE_URL}/api/{OR_REALM}/asset/attributes"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        # OpenRemote REST expects a list of attribute updates
        payload = [
            {"ref": {"id": OR_ASSET_ID, "name": "Data"}, "value": data_dict},
            {"ref": {"id": OR_ASSET_ID, "name": "notes"}, "value": ""},
            {"ref": {"id": OR_ASSET_ID, "name": "zone_snapshot_url"}, "value": data_dict.get("snapshot_url", "")}
        ]
        try:
            res = requests.put(url, json=payload, headers=headers, verify=False, timeout=10)
            if res.status_code in [200, 204]:
                pass 
            else:
                print(f"REST Sync Failed: {res.status_code}")
        except Exception as e:
            print(f"REST Sync Error: {e}")

    # 2. Sync via MQTT (Secondary) - DISABLED TO PREVENT DUPLICATES
    # global mqtt_client
    # if mqtt_client:
    #     try:
    #         mqtt_client.publish(MQTT_TOPIC, json.dumps(data_dict))
    #     except: pass

# ===================== COLOR MAPPING ==================
def get_color_name(r, g, b):
    # Standard color palette for mapping
    colors = {
        "Black": (0, 0, 0),
        "White": (255, 255, 255),
        "Red": (255, 0, 0),
        "Green": (0, 255, 0),
        "Blue": (0, 0, 255),
        "Yellow": (255, 255, 0),
        "Cyan": (0, 255, 255),
        "Magenta": (255, 0, 255),
        "Gray": (128, 128, 128),
        "Orange": (255, 165, 0),
        "Brown": (165, 42, 42),
        "Pink": (255, 192, 203),
        "Purple": (128, 0, 128)
    }
    
    # Calculate Euclidean distance to find the closest color
    min_dist = float('inf')
    closest_name = "Unknown"
    
    for name, (cr, cg, cb) in colors.items():
        dist = np.sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
        if dist < min_dist:
            min_dist = dist
            closest_name = name
            
    # Refine for dark/light variations
    brightness = (r + g + b) / 3
    if brightness < 40: return "Black" # Increased threshold slightly
    if brightness > 210: return "White"
    
    # Custom Logic for Yellow vs Orange
    # Yellow: High Red, High Green (close values), Low Blue
    # Orange: High Red, Medium Green (Red >> Green), Low Blue
    if r > 150 and g > 150 and b < 100:
        if abs(r - g) < 40: # If Red and Green are close -> Yellow
            return "Yellow"
        elif r > g: # If Red is significantly more than Green -> Orange
            return "Orange"

    return closest_name

# ===================== UTILS ==========================
def capture_zone_snapshot():
    """
    Captures a snapshot with zone visualization.
    - OLD ZONE: Drawn in RED color
    - NEW ZONE: Drawn in GREEN color
    This helps visually identify zone changes over time.
    """
    global outputFrame, zone_pts, stats
    with lock:
        if outputFrame is not None:
            snap = outputFrame.copy()
            
            # ============================================================
            # DUAL ZONE VISUALIZATION (OLD vs NEW)
            # ============================================================
            # Get the old zone from history (if exists)
            old_zone = None
            if len(stats.get("zone_history", [])) > 0:
                # Get the most recent previous zone
                old_zone = stats["zone_history"][0].get("coordinates")
            
            # Draw OLD ZONE in RED (if exists)
            if old_zone and old_zone != zone_pts:
                cv2.polylines(snap, [np.array(old_zone, np.int32)], True, (0, 0, 255), 3)  # RED
                # Add label for old zone
                if len(old_zone) > 0:
                    cv2.putText(snap, "OLD ZONE", (old_zone[0][0], old_zone[0][1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw NEW ZONE in GREEN
            cv2.polylines(snap, [np.array(zone_pts, np.int32)], True, (0, 255, 0), 3)  # GREEN
            # Add label for new zone
            if len(zone_pts) > 0:
                cv2.putText(snap, "NEW ZONE", (zone_pts[0][0], zone_pts[0][1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # ============================================================
            
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"zone_snapshot_{ts}.jpg"
            fpath = os.path.join(STATIC_DIR, fname)
            cv2.imwrite(fpath, snap)
            
            # Archive in history
            snap_data = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image": f"/static/{fname}",
                "coordinates": list(zone_pts)
            }
            stats["zone_history"].insert(0, snap_data)
            
            # Silently saved
            snap_url = f"http://{LOCAL_IP}:5000/static/{fname}"
            
            # Encode for OpenRemote
            small_snap = cv2.resize(snap, (320, 240))
            _, buffer = cv2.imencode('.jpg', small_snap, [cv2.IMWRITE_JPEG_QUALITY, 40])
            snap_b64 = base64.b64encode(buffer).decode('utf-8')
            
            stats["current_snapshot_url"] = snap_url
            stats["current_snapshot_b64"] = f"data:image/jpeg;base64,{snap_b64}"
            return snap_url
    return None

def load_stats():
    global stats
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r") as f:
                loaded = json.load(f)
                with lock:
                    # Update stats and ensure keys exist
                    for key in loaded:
                        stats[key] = loaded[key]
            # Silently loaded
        except: 
            pass # Silently ignore or start fresh

def save_stats_to_disk():
    with lock:
        try:
            with open(STATS_FILE, "w") as f:
                json.dump(stats, f, indent=4)
        except Exception as e:
            print(f"Error saving stats: {e}")

# ===================== WORKER THREAD ==================
def worker_logic():
    # print("Worker thread started.")
    while True:
        try:
            task = task_queue.get()
            if task is None: break 
            
            t_type = task.get("type")
            if t_type == "intrusion":
                process_intrusion_task(task)
            elif t_type == "exit":
                process_exit_task(task)
            elif t_type == "resume":
                process_resume_task(task)
            
            task_queue.task_done()
        except Exception as e:
            print(f"Worker Error: {e}")

def process_intrusion_task(task):
    track_id = task["track_id"]
    frame_analysis = task["frame_analysis"] 
    frame_vis = task["frame_vis"] 
    box = task["box"] 
    x1, y1, x2, y2 = box
    
    try:
        person_crop = frame_analysis[max(0,y1):y2, max(0,x1):x2]
        h, w, _ = person_crop.shape
        if h > 0 and w > 0:
            shirt_crop = person_crop[0:h//2, :]
            pant_crop = person_crop[h//2:h, :]
            
            def analyze_color(crop_arr):
                if crop_arr.size == 0: return "#808080", "Gray"
                h, w, _ = crop_arr.shape
                ch, cw = int(h*0.3), int(w*0.3)
                cc = crop_arr[ch:h-ch, cw:w-cw]
                if cc.size == 0: cc = crop_arr
                
                avg = np.average(np.average(cc, axis=0), axis=0)
                r, g, b = map(int, avg)
                # print(f"DEBUG RGB ANALYSIS: R={r} G={g} B={b}")
                hex_code = "#{:02x}{:02x}{:02x}".format(min(255,r), min(255,g), min(255,b))
                color_name = get_color_name(r, g, b)
                return hex_code, color_name

            shirt_hex, shirt_name = analyze_color(shirt_crop)
            pant_hex, pant_name = analyze_color(pant_crop)
        else:
            shirt_hex, shirt_name = "#000000", "Black"
            pant_hex, pant_name = "#000000", "Black"

        def render_label(hex_code, name):
            # Display color name and a small circle together
            return f"<div style='display:flex;align-items:center;gap:6px;'><div style='width:18px;height:18px;background-color:{hex_code};border-radius:50%;border:1px solid #fff;'></div><b style='color:#fff;text-shadow:0 0 2px #000;'>{name}</b></div>"

        shirt_display = render_label(shirt_hex, shirt_name)
        pant_display = render_label(pant_hex, pant_name)
    except Exception as e:
        print(f"Color Error: {e}")
        shirt_display, pant_display = "<span>Err</span>", "<span>Err</span>"

    # Save Image
    try:
        fname = f"ID{track_id}_{datetime.datetime.now().strftime('%H%M%S')}.jpg"
        filepath = os.path.join(CAPTURE_DIR, fname)
        cv2.imwrite(filepath, frame_vis)
        
        # Convert to Base64 for OpenRemote (Resized for reliability)
        small_img = cv2.resize(frame_vis, (320, 240))
        _, buffer = cv2.imencode('.jpg', small_img, [cv2.IMWRITE_JPEG_QUALITY, 50])
        b64_image = base64.b64encode(buffer).decode('utf-8')
    except: return

    # Update State
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log = {
        "id": f"#{track_id}", # Keep for legacy/internal lookups
        "person_id": f"Person_{track_id}",
        "photo": f"/captures/{fname}",
        "shirt": shirt_display,
        "pant": pant_display,
        "shirt_name": shirt_name,
        "pant_name": pant_name,
        "time_in": ts,
        "timestamp": ts,      # For Local Dashboard
        "time_out": "--:--:--",
        "out_time": "--:--:--" # For Local Dashboard
    }
    
    # Person detection coordinates are NOT logged to file (only zone changes are logged)
    
    with lock:
        stats["logs"].insert(0, log)
        stats["logs"] = stats["logs"][:20]
    save_stats_to_disk()

    sync_data = {
        "id": f"#{track_id}",
        "person_id": f"Person_{track_id}",
        "time_in": ts,
        "time_out": "--:--:--",
        "photo": f"http://{LOCAL_IP}:5000/captures/{fname}",
        "zonesnapshot": stats.get("current_snapshot_url", ""),
        "shirt": shirt_name,
        "pant": pant_name,
        "box_coords": [x1, y1, x2, y2]
    }
    sync_to_openremote(sync_data)

def process_exit_task(task):
    with lock:
        for l in stats["logs"]:
            if l["id"] == f"#{task['track_id']}" and l["time_out"] == "--:--:--":
                l["time_out"] = task["timestamp"]
                l["out_time"] = task["timestamp"] # For Local Dashboard
                sync_data = {
                    "id": l["id"],
                    "person_id": l["person_id"],
                    "time_in": l["time_in"],
                    "time_out": l["time_out"],
                    "photo": f"http://{LOCAL_IP}:5000{l['photo']}",
                    "zonesnapshot": stats.get("current_snapshot_url", ""),
                    "shirt": l.get("shirt_name", ""),
                    "pant": l.get("pant_name", "")
                }
                sync_to_openremote(sync_data) # Re-enabled to save Time Out
                break

def process_resume_task(task):
    with lock:
        for l in stats["logs"]:
            if l["id"] == f"#{task['track_id']}":
                l["time_out"] = "--:--:--"
                l["out_time"] = "--:--:--" # For Local Dashboard
                break

def setup_zone(picam2):
    print("\n>>> SETUP MODE: Left Click to add points, 's' to Save, 'c' to Clear, 'q' to Cancel <<<")
    new_pts = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            new_pts.append([x, y])
            print(f"   + Point added: ({x}, {y})")

    win_name = "SET INTRUSION ZONE (Click to Draw)"
    try:
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, mouse_callback)
    except: pass

    while True:
        raw = picam2.capture_array()
        if raw is None: continue
        # picamera2 BGR -> RGB for display if needed, but we typically use BGR for vis
        vis_setup = raw.copy()
        
        # Draw points and polygon
        for i, p in enumerate(new_pts):
            cv2.circle(vis_setup, tuple(p), 4, (0, 255, 255), -1)
            cv2.putText(vis_setup, str(i+1), (p[0]+5, p[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
            
        if len(new_pts) > 1:
            cv2.polylines(vis_setup, [np.array(new_pts, np.int32)], False, (0, 255, 255), 2)
        if len(new_pts) > 2:
            cv2.polylines(vis_setup, [np.array(new_pts, np.int32)], True, (0, 255, 255), 2)

        cv2.putText(vis_setup, f"POINTS: {len(new_pts)}/8+", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(vis_setup, "'s':Save, 'c':Clear, 'q':Cancel", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        # Sync with Web Stream
        global outputFrame
        with lock:
            outputFrame = vis_setup.copy()
            
        try:
            cv2.imshow(win_name, vis_setup)
        except: pass
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and len(new_pts) >= 3:
            # Save points
            with open(CONFIG_FILE, "w") as f:
                json.dump(new_pts, f)
            
            # ============================================================
            # ZONE COORDINATES LOGGING - PYTHON SETUP MODE
            # ============================================================
            # This saves the zone coordinates to intrusion_coords.txt when:
            # - User draws a zone using mouse clicks in OpenCV window
            # - User presses 's' key to save the zone
            # Format: TIMESTAMP | ZONE_UPDATE | Points:[[x1,y1], [x2,y2], ...]
            # This creates a permanent history of all zones (old to new)
            # ============================================================
            try:
                with open(COORDS_LOG_FILE, "a") as f:
                    f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ZONE_UPDATE | Points:{new_pts}\n")
            except: pass
            
            # SAVE SCREENSHOT OF ZONE
            try:
                snap_path = capture_zone_snapshot()
                if snap_path:
                    with lock:
                        stats["reference_zone"] = snap_path
            except Exception as e:
                print(f"Error saving zone snapshot: {e}")

            try: cv2.destroyWindow(win_name)
            except: pass
            return new_pts
        elif key == ord('c'):
            new_pts = []
        elif key == ord('q') or key == 27:
            try: cv2.destroyWindow(win_name)
            except: pass
            return None

@app.route('/api/cmd/<command>')
def remote_command(command):
    global reset_flag, shutdown_flag, stats
    if command == 'r':
        global reset_flag
        reset_flag = True
        return jsonify({"status": "Reset triggered. Historical data preserved. New zone setup active."})
    elif command == 'toggle_zone':
        with lock:
            stats["zone_active"] = not stats.get("zone_active", True)
            status = "ACTIVE" if stats["zone_active"] else "INACTIVE"
        return jsonify({"status": f"Zone {status}"})
    elif command == 'q':
        shutdown_flag = True
        return jsonify({"status": "Shutting down"})
    return jsonify({"status": "Unknown command"}), 400

# ===================== FLASK ROUTES ===================
@app.route('/')
def index():
    # Priority for intrusion1.html which has the web interface
    paths = [os.path.join(BASE_DIR, "intrusion1.html"), os.path.join(BASE_DIR, "templates", "intrusion1.html"), os.path.join(BASE_DIR, "intrusion.html")]
    for p in paths:
        if os.path.exists(p):
            with open(p, "r") as f: return f.read()
    return "Error: HTML template missing.", 500

@app.route('/captures/<path:filename>')
def serve_capture(filename):
    from flask import send_from_directory
    return send_from_directory(CAPTURE_DIR, filename)

@app.route('/api/data')
def get_data():
    with lock: return jsonify(stats)

@app.route('/api/save_zone', methods=['POST'])
def save_zone():
    global zone_pts, current_zone
    try:
        new_pts = request.json.get("points")
        if new_pts and len(new_pts) >= 3:
            with zone_lock:
                zone_pts = new_pts
                current_zone = Polygon(zone_pts)
                with open(CONFIG_FILE, "w") as f:
                    json.dump(zone_pts, f)
                
                # ============================================================
                # ZONE COORDINATES LOGGING - WEB BROWSER MODE
                # ============================================================
                # This saves the zone coordinates to intrusion_coords.txt when:
                # - User draws a zone on the web dashboard interface
                # - User clicks "Save New Zone" button
                # Format: TIMESTAMP | ZONE_UPDATE_WEB | Points:[[x1,y1], [x2,y2], ...]
                # This creates a permanent history of all zones (old to new)
                # Note: Detection coordinates are NOT saved here (only zone boundaries)
                # ============================================================
                try:
                    with open(COORDS_LOG_FILE, "a") as f:
                        f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ZONE_UPDATE_WEB | Points:{zone_pts}\n")
                except: pass
                
                # Capture reference frame for web save
                snap_img = capture_zone_snapshot()
                
                # Sync Zone to OpenRemote
                sync_to_openremote({
                    "zone_coordinates": zone_pts,
                    "snapshot_url": snap_img if snap_img else "", # Match sync_to_openremote key
                    "zone_snapshot_url": snap_img if snap_img else ""
                })

                save_stats_to_disk()

            print(">>> Zone updated from Web Browser! <<<")
            return jsonify({"success": True})
    except Exception as e:
        print(f"Web Setup Error: {e}")
    return jsonify({"success": False}), 400

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            encodedImage = None
            with lock:
                if outputFrame is not None:
                    (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
                    if not flag: encodedImage = None
            if encodedImage is None:
                time.sleep(0.1); continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            time.sleep(0.04)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ===================== MAIN LOOP ======================
if __name__ == "__main__":
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
    if not os.path.exists(COORDS_LOG_FILE):
        with open(COORDS_LOG_FILE, "w") as f: f.write("ZONE COORDINATES LOG\n====================\n")
    
    load_stats() # Load previous database/stats
    init_mqtt() # Initialize MQTT Connection
    
    # Send System Online Notification (Snapshot only)
    sync_to_openremote({
        "zone_snapshot_url": stats.get("current_snapshot_url", "")
    })

    # Raspberry Pi Camera Module Setup
    print("[INFO] Initializing Pi Camera...", flush=True)
    try:
        from picamera2 import Picamera2
        
        # Check for available cameras first
        cameras = Picamera2.global_camera_info()
        print(f"[INFO] Detected {len(cameras)} camera(s)", flush=True)
        
        if len(cameras) == 0:
            print("[ERROR] No cameras detected!", flush=True)
            print("[FIX] Please check:", flush=True)
            print("  1. Camera ribbon cable is properly connected", flush=True)
            print("  2. Camera is enabled: sudo raspi-config → Interface Options → Camera", flush=True)
            print("  3. Run: libcamera-hello --list-cameras", flush=True)
            sys.exit(1)
        
        # Explicitly use first camera (index 0)
        picam2 = Picamera2(0)
        config = picam2.create_video_configuration(main={"size": (640, 480), "format": "BGR888"})
        picam2.configure(config)
        picam2.start()
        print("[INFO] Camera started successfully!", flush=True)
        
    except IndexError as idx_err:
        print(f"[ERROR] Camera index error: {idx_err}", flush=True)
        print("[FIX] No camera found at index 0. Check camera connection.", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Pi Camera failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    threading.Thread(target=worker_logic, daemon=True).start()
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, threaded=True), daemon=True).start()
    
    # Automatically open local dashboard (if display exists)
    print(f"[INFO] System Active. Dashboard: http://{os.popen('hostname -I').read().split()[0] if os.name != 'nt' else '127.0.0.1'}:5000", flush=True)
    time.sleep(2)
    try: webbrowser.open("http://127.0.0.1:5000")
    except: pass

    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f: 
            zone_pts = json.load(f)
            current_zone = Polygon(zone_pts)
        print(">>> Loaded existing zone configuration. <<<")
    else: 
        print(">>> No zone config found. Using default. <<<")
    
    print("[INFO] Loading YOLO model...", flush=True)
    model = YOLO("best_ncnn_model", task="detect")
    print("[INFO] Model Loaded Successfully.", flush=True)
    
    # Ensure we have a snapshot for the Zone URL
    if not stats.get("current_snapshot_url"):
        capture_zone_snapshot()
    
    logged_ids, active = set(), {}
    zone_timers = {}
    COORD_LOG = os.path.join(BASE_DIR, "intruder_movement.log")
    f_start, f_cnt, cur_fps = time.time(), 0, 0.0
    
    # Updated Filters for robust detection
    MIN_CONF = 0.50  # Balanced confidence
    MIN_AREA = 3500  # Filter out floor noise/small artifacts

    print(">>> SYSTEM ACTIVE: HEADLESS WEB-ONLY MODE <<<")

    try:
        while True:
            if shutdown_flag: break
            
            # Pi Camera Capture (Native BGR)
            raw = picam2.capture_array()
            if raw is None:
                time.sleep(0.01)
                continue

            # Corrected Color Pipeline:
            # If Red was Blue, it means the raw data was coming in as RGB 
            # while the display expects BGR.
            vis = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR) 
            frame_rgb = raw.copy() # frame_rgb is now true RGB for YOLO/Theory
            
            results = model.track(frame_rgb, persist=True, conf=MIN_CONF, imgsz=320, verbose=False, tracker="bytetrack.yaml")
            
            for r in results:
                if r.boxes is None: continue
                for b in r.boxes:
                    if int(b.cls[0]) == 0 and b.id is not None:
                        # Box Coordinates
                        coords = b.xyxy[-1] if hasattr(b.xyxy, 'shape') and len(b.xyxy.shape) > 1 else b.xyxy[0]
                        tid, x1, y1, x2, y2 = int(b.id[0]), *map(int, coords)
                        
                        w, h = x2 - x1, y2 - y1
                        # Filter floor/small noise
                        if w * h < MIN_AREA: continue
                        
                        # POSITION FLEXIBILITY: Removed strict aspect ratio check.
                        # Now detects people in any position (sitting, crawling, etc.)
                        
                        with zone_lock:
                            inside = current_zone.contains(Point((x1+x2)/2, y2))
                        
                        col = (0, 255, 0) # Green
                        
                        if inside:
                            with lock: zone_is_active = stats.get("zone_active", True)
                            
                            if zone_is_active:
                                col = (0, 0, 255) # RED - Instant Intrusion
                                
                                # Log Coordinates
                                try:
                                    with open(COORD_LOG, "a") as f:
                                        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        # Box format: [x1, y1, x2, y2]
                                        f.write(f"{ts} | ID:{tid} | Box:[{x1},{y1},{x2},{y2}]\n")
                                except: pass
                                
                                # Trigger Alert (Strictly Once per ID)
                                if tid not in logged_ids:
                                    logged_ids.add(tid)
                                    with lock:
                                        stats["total_intrusions"] += 1
                                        stats["alerts_sent"] += 1
                                        stats["last_breach"] = datetime.datetime.now().strftime("%H:%M:%S")
                                    
                                    task_queue.put({
                                        "type":"intrusion", 
                                        "track_id":tid, 
                                        "frame_analysis":frame_rgb.copy(), 
                                        "frame_vis":vis.copy(), 
                                        "box":[x1,y1,x2,y2]
                                    })
                            else:
                                col = (255, 165, 0) # Orange (Disarmed)
                        
                        # Dynamic Label based on state
                        label = f"ID:{tid}"
                        if inside and zone_is_active:
                            label = f"INTRUDER {tid}"
                        elif inside:
                            label = f"DETECTED {tid}"
                        cv2.rectangle(vis, (x1,y1), (x2,y2), col, 2)
                        cv2.putText(vis, label, (x1,y1-10), 0, 0.5, col, 2)
                        
                        if inside: active[tid] = datetime.datetime.now()
                        else:
                            if tid in active: active[tid] = datetime.datetime.now()

            now = datetime.datetime.now()
            for tid in list(active.keys()):
                if (now - active[tid]).total_seconds() > 3.0:
                    task_queue.put({"type":"exit", "track_id":tid, "timestamp":now.strftime("%H:%M:%S")})
                    del active[tid]

            with zone_lock:
                display_pts = zone_pts
                
            with lock:
                zone_is_active = stats.get("zone_active", True)
            
            if zone_is_active:
                cv2.polylines(vis, [np.array(display_pts, np.int32).reshape((-1,1,2))], True, (0,255,255), 2)
            
            f_cnt += 1
            if time.time() - f_start >= 1.0:
                cur_fps = f_cnt / (time.time() - f_start)
                f_start, f_cnt = time.time(), 0
                with lock:
                    stats["fps"] = round(cur_fps, 1)
                    uptime_sec = int(time.time() - stats["system_start"])
                    stats["uptime"] = str(datetime.timedelta(seconds=uptime_sec))
            
            cv2.putText(vis, f"FPS: {cur_fps:.1f}", (10,30), 0, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, "Press 'r' to Reset Zone", (10, 60), 0, 0.6, (0, 255, 255), 2)

            with lock: outputFrame = vis.copy()

            # if f_cnt % 30 == 0:
            #     ts = datetime.datetime.now().strftime("%H:%M:%S")
            #     print(f"[{ts}] Monitoring Active. FPS: {cur_fps:.1f} | Detections in Frame: {len(results[0].boxes) if results else 0}", flush=True)

            # Headless mode: Reset handled by Web UI 'Draw' button
            if reset_flag:
                reset_flag = False
                logged_ids.clear()
                active.clear()
                res = setup_zone(picam2)
                if res:
                    with zone_lock:
                        zone_pts = res
                        current_zone = Polygon(zone_pts)
            
            time.sleep(0.01)

    except Exception as e:
        import traceback
        print(f"\n[CRITICAL ERROR] The system crashed: {e}")
        traceback.print_exc()
    finally:
        print("[INFO] Shutting down camera...")
        try: picam2.stop()
        except: pass
        print("[INFO] System Stopped.")
