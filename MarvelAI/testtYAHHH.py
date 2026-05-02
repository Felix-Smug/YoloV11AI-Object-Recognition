from ultralytics import YOLO
import numpy as np
import time
import keyboard
import pyautogui
import math
import torch
import bettercam
import sys

# Import the MouseInstruct class from your provided file
from mouse_instruct import MouseInstruct, DeviceNotFoundError

# Initialize MouseInstruct (Replaces Serial connection)
print("Searching for HID Mouse...")
try:
    # getMouse looks for the specific Ping Code (0xf9) defined in the firmware
    mouse = MouseInstruct.getMouse()
    print("[+] Mouse Instruct Device found!")
except DeviceNotFoundError as e:
    print(f"[-] Error: {e}")
    print("Ensure the Arduino is plugged in and the correct firmware is uploaded.")
    sys.exit()

model = YOLO("C:/Users/Smugg/Documents/Github Repos/YoloV11AI/MarvelAI/Marvel.engine")

if torch.cuda.is_available():
    print(f"Using GPU with TensorRT: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available")

class_names = ['Friendly', 'enemy']

camera = bettercam.create(output_color='BGR')

screen_w, screen_h = pyautogui.size()
region_size = 100
region_left = screen_w // 2 - region_size // 2
region_top = screen_h // 2 - region_size // 2

print("Press 'C' to toggle Aim Assist ON/OFF")
print("Press 'End' to Stop")  # Changed empty string to 'End' for safety
aim_assist_enabled = True
last_toggle_time = 0

frame_count = 0
start_time = time.time()

# Removed Serial Queue/Thread. HID writing is fast enough to happen inline.

def send_mouse_move(dx, dy):
    # MouseInstruct supports 16-bit integers (-32767 to 32767)
    # We no longer need to clamp to 127 like standard Arduino Mouse
    # However, for safety/game logic, you might still want limits.
    
    # Cast to int is required for bitwise operations in mouse_instruct.py
    ix = int(dx)
    iy = int(dy)
    
    # Optional: Safety clamp to prevent spinning if detection goes wild
    # ix = max(min(ix, 500), -500)
    # iy = max(min(iy, 500), -500)

    mouse.move(ix, iy)

def send_fire():
    # Uses MouseInstruct click method
    mouse.click()

while not keyboard.is_pressed('end'):    
    if keyboard.is_pressed('c') and time.time() - last_toggle_time > 0.5:
        aim_assist_enabled = not aim_assist_enabled
        status = "ENABLED" if aim_assist_enabled else "DISABLED"
        print(f"Aim Assist {status}")
        last_toggle_time = time.time()

    frame = camera.grab(region=(region_left, region_top, region_left + region_size, region_top + region_size))
    if frame is None:
        continue

    frame_np = np.array(frame)

    results = model.predict(
        source=frame_np,
        conf=0.35,
        verbose=False,
        device=0
    )

    if aim_assist_enabled and results and results[0].boxes is not None:
        closest_target = None
        min_distance = float('inf')

        for box in results[0].boxes:
            cls = int(box.cls[0])
            # Check range to avoid index errors if class_names isn't perfectly synced
            if cls < len(model.names):
                label = model.names[cls]
            else:
                label = str(cls)

            # Update this string to match your specific model label
            if label == "enemy_head" or label == "enemy": 
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2

                dx = box_center_x - region_size / 2
                dy = box_center_y - region_size / 2
                
                # Optimized distance calculation (squared Euclidean)
                distance = dx*dx + dy*dy 

                if distance < min_distance:
                    min_distance = distance
                    closest_target = (dx, dy)

        if closest_target:              
            dx, dy = closest_target

            # Sensitivity modifiers (Adjust as needed)
            if abs(dx) > 20:
                dx *= 1.0
            else:
                dx *= 1.0
            if abs(dy) > 20:
                dy *= 1.0
            else:
                dy *= 1.0

            send_mouse_move(dx, dy)

            # Trigger bot logic
            #if abs(dx) < 3 and abs(dy) < 3:
            #    send_fire()

    frame_count += 1
    if frame_count >= 60:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = time.time()

print("Stopped")