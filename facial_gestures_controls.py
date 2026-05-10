
# IMPORTS 

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time




# GLOBAL STATES

curr_x, curr_y = 0, 0
center_x, center_y = 0.5, 0.5  
screen_w, screen_h = pyautogui.size()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)



# SETTINGS

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
FRAME_BOUNDARY = 0.12  # Range around the center (smaller = faster mouse)
SMOOTHING = 0.15        # Lower = smoother but laggier
CLICK_DELAY = 0.5
blink_threshold = 0.012
SCROLL_SPEED = 25        # How many units to scroll per frame
LONG_BLINK_TIME = 0.5    # Minimum time to trigger/toggle
scroll_mode = 1          # 1 for Up, -1 for Down
right_blink_start = None
last_blink_end_time = 0


EYEBROW_SENSITIVITY = 0.02  # Adjust based on your face
last_zoom_time = 0
ZOOM_COOLDOWN = 0.3
resting_brow_dist = 0.14

# CAMERA SETTINGS
url = "http://192.168.100.9:8080/video"  
cam = cv2.VideoCapture(url) 



print("System Ready.")
print("Press 'c' to Calibrate/Recenter the cursor to your current head position.")
print("Press 'q' to Quit.")





# LOGIC 


while True:

    success, frame = cam.read()
    
    if not success or frame is None:
        print("Waiting for camera...")
        continue



    # FRAME PROCESSING

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)



    # CURSOR CONTROL


    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        nose = landmarks[1]


        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):
            center_x = nose.x
            center_y = nose.y
            print(f"Recalibrated! New Center: ({center_x:.2f}, {center_y:.2f})")


        min_x, max_x = center_x - FRAME_BOUNDARY, center_x + FRAME_BOUNDARY
        min_y, max_y = center_y - FRAME_BOUNDARY, center_y + FRAME_BOUNDARY

        scale_x = np.interp(nose.x, [min_x, max_x], [0, 1])
        scale_y = np.interp(nose.y, [min_y, max_y], [0, 1])

        target_x = scale_x * screen_w
        target_y = scale_y * screen_h

        curr_x = curr_x + (target_x - curr_x) * SMOOTHING
        curr_y = curr_y + (target_y - curr_y) * SMOOTHING
        
        pyautogui.moveTo(int(curr_x), int(curr_y))


        # CONTROLS


        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
        
            
            left_eye_top = landmarks[386].y
            left_eye_bottom = landmarks[374].y
            eye_distance = left_eye_bottom - left_eye_top

            
            if eye_distance < blink_threshold:
                if blink_start_time is None:
                    
                    blink_start_time = time.time()
                else:
                    
                    elapsed = time.time() - blink_start_time
                    if elapsed >= CLICK_DELAY:
                        pyautogui.click()
                        print("Intentional Click Triggered!")
                        
                        blink_start_time = None 
                        pyautogui.sleep(0.3) 
            else:
                
                blink_start_time = None
        



        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
        
           
            right_eye_top = landmarks[159].y
            right_eye_bottom = landmarks[145].y
            right_eye_dist = right_eye_bottom - right_eye_top

           
            if right_eye_dist < 0.012:  
                if right_blink_start is None:
                    right_blink_start = time.time()
            
                elapsed = time.time() - right_blink_start
            
                
                if elapsed > LONG_BLINK_TIME:
                    pyautogui.scroll(SCROLL_SPEED * scroll_mode)
                
            else:  
                if right_blink_start is not None:
                    total_closed_time = time.time() - right_blink_start
                
                    
                    if total_closed_time > LONG_BLINK_TIME:
                        scroll_mode *= -1  # Flip 1 to -1 or vice versa
                        direction_text = "UP" if scroll_mode == 1 else "DOWN"
                        print(f"Scroll Direction Toggled to: {direction_text}")
                
                    right_blink_start = None



        if results.multi_face_landmarks:

            
            landmarks = results.multi_face_landmarks[0].landmark
            
            brow_height = (landmarks[52].y + landmarks[282].y) / 2
            nose_y = landmarks[1].y
            brow_dist = nose_y - brow_height # Distance grows as brows lift
            if cv2.waitKey(1) & 0xFF == ord('d'):
                resting_brow_dist = brow_dist
                
       
            if time.time() - last_zoom_time > ZOOM_COOLDOWN:
              
                if brow_dist < (resting_brow_dist - EYEBROW_SENSITIVITY):
                    pyautogui.hotkey('ctrl', '+')
                    print("Zooming In (+)")
                    last_zoom_time = time.time()
            
              
                elif brow_dist > (resting_brow_dist + EYEBROW_SENSITIVITY):
                    pyautogui.hotkey('ctrl', '-')
                    print("Zooming Out (-)")
                    last_zoom_time = time.time()            



    

    current_dir = "UP" if scroll_mode == 1 else "DOWN"
    color = (0, 255, 0) if scroll_mode == 1 else (0, 0, 255)
    cv2.putText(frame, f"Scroll Mode: {current_dir}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


    cv2.imshow('Nose Pointer Control', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()