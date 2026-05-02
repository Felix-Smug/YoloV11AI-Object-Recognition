from ultralytics import YOLO
import cv2
import numpy as np
import time
import keyboard
import pyautogui
import math
import torch
import bettercam
import serial
import threading
from queue import Queue

model = YOLO("C:/Users/Smugg/Documents/Github Repos/YoloV11AI/MarvelAI/Marvel.engine")

arduino = serial.Serial('COM5', 115200, timeout=1)

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
print("Press '`' to Stop")

aim_assist_enabled = True
last_toggle_time = 0

frame_count = 0
start_time = time.time()
last_fire_time = 0
fire_cooldown = 0.2 

serial_queue = Queue()

def serial_worker():
    while True:
        command = serial_queue.get()
        if command is None:
            break
        arduino.write(command.encode())

serial_thread = threading.Thread(target=serial_worker, daemon=True)
serial_thread.start()

def send_mouse_move(dx, dy):
    dx = max(min(int(dx), 127), -127)
    dy = max(min(int(dy), 127), -127)
    command = f"MOVE {dx} {dy}\n"
    serial_queue.put(command)

def send_fire():
    serial_queue.put("FIRE\n")

while not keyboard.is_pressed('`'):
    
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
            label = model.names[cls]

            if label == "enemy_head":
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2

                dx = box_center_x - region_size / 2
                dy = box_center_y - region_size / 2
                #distance = math.sqrt(dx**2 + dy**2)
                distance = dx*dx + dy*dy 

                if distance < min_distance:
                    min_distance = distance
                    closest_target = (dx, dy)

        if closest_target:              
            dx, dy = closest_target

            if abs(dx) > 20:
                dx *= 1.0
            else:
                dx *= 1.0
            if abs(dy) > 20:
                dy *= 1.0
            else:
                dy *= 1.0

            send_mouse_move(dx, dy)

            if abs(dx) < 3 and abs(dy) < 3:
                send_fire()

    frame_count += 1
    if frame_count >= 60:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = time.time()

serial_queue.put(None)
serial_thread.join()

print("Stopped")
