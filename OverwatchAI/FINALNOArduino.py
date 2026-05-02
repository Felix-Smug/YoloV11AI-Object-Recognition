from ultralytics import YOLO
import cv2
import numpy as np
import time
import keyboard
import pyautogui
import math
import torch
import bettercam
import win32api
import win32con

model = YOLO("C:/Users/Smugg/Documents/Github Repos/YoloV11AI/OverwatchAI/ow2epoch60.engine")


if torch.cuda.is_available():
    print(f"Using GPU with TensorRT: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available")

class_names = ['EnemyHead']

camera = bettercam.create(output_color='BGR')

screen_w, screen_h = pyautogui.size()
region_size = 130
region_left = screen_w // 2 - region_size // 2
region_top = screen_h // 2 - region_size // 2

print("Press 'B' to toggle Aim Assist ON/OFF")
print("Press '`' to Stop")

aim_assist_enabled = True
last_toggle_time = 0

frame_count = 0
start_time = time.time()


def click():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)

while not keyboard.is_pressed('`'):
    
    if keyboard.is_pressed('b') and time.time() - last_toggle_time > 0.5:
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
        conf=0.2,
        verbose=False,
        device=0
    )

    if aim_assist_enabled and results and results[0].boxes is not None:
        closest_target = None
        min_distance = float('inf')

        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "EnemyHead":
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2

                dx = box_center_x - region_size / 2
                dy = box_center_y - region_size / 2
                distance = math.sqrt(dx**2 + dy**2)

                if distance < min_distance:
                    min_distance = distance
                    closest_target = (dx, dy)

        if closest_target:
            dx, dy = closest_target

            if abs(dx) > 30:
                dx *= 1.2
            else:
                dx *= 1.2
            if abs(dy) > 20:
                dy *= 1.2
            else:
                dy *= 1.2

            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)

            #remove these two lines for just aim assist
            #if abs(dx) < 3 and abs(dy) < 3:
                #click()
           
    frame_count += 1
    if frame_count >= 60:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = time.time()

print("Stopped")
