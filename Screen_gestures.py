import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
import pyautogui # Python lib to control mouse and keyboard

model_path = "hand_landmarker.task"
# pyautogui.FAILSAFE = False

# Aliases
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_landmarks = []

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_landmarks
    print('Hand landmarker result: {}'.format(result))
    # Object 'result' contains 3 attributes - handedness, hand_landmarks, hand_world_landmarks
    latest_landmarks = result.hand_landmarks # list of lists

def dis(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def dis2(p1, l):
    return math.hypot(p1.x - l[0], p1.y - l[1])

x_history = []
y_history = []
z_history = []

thumb_his = []

# drag_start_x = 0
# drag_start_y = 0

# drag_end_x = 0
# drag_end_y = 0

# z_max_change = 0

# Cooldown times
last_click_time = 0
click_cooldown = 0.5

PINCH_ON = 0.20
PINCH_OFF = 0.40

last_scroll_time = 0
scroll_cooldown = 0.1

# last_drag_time = 0
# drag_cooldown = 0.6

def checkGesture(hand: list, pinching: bool):
    w = hand[0]
    x = hand[8].x
    y = hand[8].y
    z = hand[8].z

    thumb = hand[4]

    global x_history
    global y_history
    global z_history

    global thumb_his
    
    x_history.append(x)
    y_history.append(y)
    z_history.append(z)

    thumb_his.append(thumb)

    if len(z_history) >= 40:
        z_history.pop(0)

    # cursor movement (index finger open)
    if (
        (dis(w, hand[20]) < dis(w, hand[17])) and 
        (dis(w, hand[16]) < dis(w, hand[13])) and 
        (dis(w, hand[12]) < dis(w, hand[9])) and 
        (1.2*dis(hand[4], hand[10]) < dis(w, hand[4])) and 
        (dis(hand[8], w) > dis(w, hand[5])) and
        (not pinching)
        ):
        # clicking (change in depth z)
        if (len(z_history) >= 16 and z_history[-15]-z_history[-7]>=0.05 and z_history[-7]-z_history[-1]<=-0.05):
            # global z_max_change
            # z_max_change = max(z_max_change, z_history[-21]-z_history[-11])
            return 2
        else:
            return 1
    elif (
        (dis(w, hand[20]) < dis(w, hand[17])) and 
        (dis(w, hand[16]) < dis(w, hand[13])) and 
        (dis(w, hand[12]) > dis(w, hand[9])) and 
        (1.2*dis(hand[4], hand[10]) < dis(w, hand[4])) and 
        (dis(hand[8], w) > dis(w, hand[5])) and
        (dis(hand[6], hand[10]) <= 1.2*dis(hand[5], hand[9]))
        ):
        return 3

    else:
        return -1
    # elif (
    #     (dis(hand[4], hand[8]) < 0.15)
    # ):
    #     return 4
    
       

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

with HandLandmarker.create_from_options(options) as landmarker:
    
    cap = cv2.VideoCapture(0)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print("Camera FPS:", fps)
    smooth_x, smooth_y = 0, 0
    old_15_x, old_15_y = 0, 0
    old_5_y = 0

    dragging = False
    pinching = False

    while cap.isOpened():
        
        success, frame = cap.read()

        if not success:
            break

        h, w = frame.shape[:2]
        print(f"Height: {h}, Width: {w}")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

        landmarker.detect_async(mp_image, timestamp_ms)
        """
        Understanding the actual output
        - Landmark just denotes a point, implying (x, y, z)
        - Same 21 points are represented in 2 coordinate systems - NormalizedLandmark and Landmark
        - hand_landmarks shows points in NormalizedLandmark
            Here, each (x,y) of each point is normalized to a value between 0 and 1
            z indicates closeness to the camera (0 being wrist landmark, -ve is closer, +ve is farther)
        - hand_world_landmarks shows points in Landmark
            x, y, z are in real-world units of meters, with origin at rough center of palm
            Removes ambiguities introduced from variable distances from camera (values remain same)
        """

        for hand in latest_landmarks: # to compute per hand
            # pinch distance relative to hand size (wrist -> middle finger base),
            # so the threshold doesn't depend on how far the hand is from the camera
            hand_size = dis(hand[0], hand[9])
            pinch_ratio = dis(hand[4], hand[8]) / hand_size if hand_size > 0 else 1

            # hysteresis: start pinching below PINCH_ON, stop only above PINCH_OFF
            # (no dead zone in between, and no flicker around a single threshold)
            if pinching:
                pinching = pinch_ratio < PINCH_OFF
            else:
                pinching = pinch_ratio < PINCH_ON

            gesture = checkGesture(hand, pinching)
            tip = hand[8]
            x = 1 - tip.x
            y = tip.y

            # if gesture == 3:
            #     y = 1-tip.y

            if (len(x_history)>= 16):
                old_15_x = 1-x_history[-15]
                old_15_y = y_history[-15]

            x_min, x_max = 0.1, 0.9
            y_min, y_max = 0.1, 0.8

            screen_width, screen_height = pyautogui.size()

            # needed since x and y are normalized, and to add a padding
            screen_x = (x - x_min) / (x_max - x_min) * screen_width
            screen_y = (y - y_min) / (y_max - y_min) * screen_height

            old_screen_x = (old_15_x - x_min) / (x_max - x_min) * screen_width
            old_screen_y = (old_15_y - y_min) / (y_max - y_min) * screen_height

            if (len(y_history)>=5):
                # old_5_y = (y_history[-4] - y_min) / (y_max - y_min) * screen_height
                old_5_y = (y_history[-3] - y_min) / (y_max - y_min) * screen_height

            # exponential smoothening of cursor movement - 0.3 smoothening factor
            smooth_x = 0.7*smooth_x + 0.3*screen_x
            smooth_y = 0.7*smooth_y + 0.3*screen_y

            flag = 0

            if gesture == -1 and pinching:
                if not dragging:
                    pyautogui.moveTo(smooth_x, smooth_y)
                    pyautogui.mouseDown()
                    dragging = True
                pyautogui.moveTo(smooth_x, smooth_y)
                flag += 1

            elif dragging and not pinching:
                pyautogui.moveTo(smooth_x, smooth_y)
                pyautogui.mouseUp()
                dragging = False
                flag += 1

            elif gesture == 1:
                pyautogui.moveTo(smooth_x, smooth_y) # moves the cursor
                flag += 1

            elif gesture == 2:
                # pyautogui.moveTo(old_screen_x, old_screen_y) # moves the cursor
                # pyautogui.click(old_screen_x, old_screen_y) # clicks (left click)
                
                current_time = time.time()
                if (current_time-last_click_time >= click_cooldown):
                    pyautogui.click(old_screen_x, old_screen_y) # clicks (left click)
                    last_click_time = current_time
                pyautogui.moveTo(screen_x, screen_y) # moves the cursor
                flag += 1

            elif gesture == 3:
                current_time = time.time()
                if (len(y_history)>=5 and current_time-last_scroll_time >= scroll_cooldown):
                    # pyautogui.scroll(-60*max(0, math.ceil(y_history[-8]-y_history[-1])))
                    # pyautogui.scroll(math.floor(round((0.3*(old_5_y-smooth_y)), -1)))
                    # pyautogui.scroll(math.floor(0.05*(old_5_y-screen_y)))
                    dy = screen_y - old_5_y

                    if abs(dy) > 7:
                        scroll_amount = round(dy / 5)
                        # scroll_amount = np.clip(scroll_amount, -10, 10)

                        pyautogui.scroll(int(scroll_amount))
                    last_scroll_time = current_time
                flag += 1

            # elif gesture == 4:
            #     # time.sleep(3)
            #     # current_time = time.time()
            #     if (len(thumb_his) >= 8):
            #         if (dis2(thumb_his[-2], [x_history[-2], y_history[-2]]) > 0.05 and dis(hand[4], hand[8]) < 0.05):
            #             pyautogui.mouseDown()
            #         elif (dis2(thumb_his[-2], [x_history[-2], y_history[-2]]) < 0.05 and dis(hand[4], hand[8]) > 0.05):
            #             pyautogui.mouseUp()
            #         # else:
            #             # pyautogui.moveTo(screen_x, screen_y)


                # if (drag_start_x == 0 and drag_start_y == 0):
                #     drag_start_x = screen_x
                #     drag_start_y = screen_y
                # else:
                #     drag_end_x = screen_x
                #     drag_end_y = screen_y

                # if (current_time-last_drag_time >= drag_cooldown):
                #     pyautogui.dragTo(screen_x, screen_y, duration=0.75)
                #     last_drag_time = current_time

                # pyautogui.dragTo(screen_x, screen_y)

                # flag += 1

            if flag > 0:
                cv2.putText(frame, f"{screen_x}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"{screen_y}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"{x}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"{y}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                h, w, _ = frame.shape

                px = int(tip.x * w)
                py = int(tip.y * h)

                cv2.circle(frame, (px, py), 10, (0, 0, 255), -1)

        
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # print("Maximum change in z: ", z_max_change)
            break

    cap.release()
    cv2.destroyAllWindows()
