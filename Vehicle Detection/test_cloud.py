import paho.mqtt.client as mqtt
import time
import json

# CREDENTIALS
BROKER = "109.176.197.144"
PORT = 1883
USER = "master:new_user"
PASS = "p6nwLTG02ZRfjbMiNwDYzgGZd1G7OVmh"
ASSET_ID = "4vC1DFDuGDdd44gB6z9D5B"

# CRITICAL: Client ID MUST match the username (without realm)
CLIENT_ID = "new_user" 

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("✅ CONNECTED to OpenRemote!")
        
        # TEST PAYLOAD
        payload = json.dumps({"status": "TEST_SUCCESS", "timestamp": time.time()})
        
        # --- STRATEGY: BLAST ALL 3 COMMON TOPIC FORMATS ---
        
        # Topic 1: Standard Service User Format
        t1 = f"master/{CLIENT_ID}/writeattributevalue/data/{ASSET_ID}"
        print(f"🔫 Firing at Topic 1: {t1}")
        client.publish(t1, payload)
        
        # Topic 2: Direct Asset Format (Sometimes used for admins)
        t2 = f"master/{ASSET_ID}/attribute/data"
        print(f"🔫 Firing at Topic 2: {t2}")
        client.publish(t2, payload)
        
        # Topic 3: Old Style
        t3 = f"writeattributevalue/data/{ASSET_ID}"
        print(f"🔫 Firing at Topic 3: {t3}")
        client.publish(t3, payload)
        
        print("\n⚡ PACKETS SENT. Check Dashboard for 'TEST_SUCCESS'.")
        client.disconnect()
    else:
        print(f"❌ Connection Failed. Code: {rc}")

client = mqtt.Client(client_id=CLIENT_ID, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(USER, PASS)
client.on_connect = on_connect

print("Connecting...")
client.connect(BROKER, PORT, 60)
client.loop_forever()
