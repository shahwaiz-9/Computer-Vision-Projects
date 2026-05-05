import cv2
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))



pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SENSITIVITY_X = 35.0  # Separate X and Y for better control
SENSITIVITY_Y =  SENSITIVITY_X * 1.6
SMOOTHING = 0.05


calibrated = False
base_x, base_y = 0, 0
curr_x, curr_y = pyautogui.size()[0]//2, pyautogui.size()[1]//2



url = "http://192.168.100.9:8080/video"  

cam = cv2.VideoCapture(url)

face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)


screen_w, screen_h = pyautogui.size()



while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    ih, iw, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        iris = landmarks[474]
        anchor = landmarks[6] 

 
        if not calibrated:
            base_x = iris.x - anchor.x
            base_y = iris.y - anchor.y
            calibrated = True
            print("Calibrated! Look at the center of your screen.")


        offset_x = -((iris.x - anchor.x) - base_x)
        offset_y = -((iris.y - anchor.y) - base_y)

    
        target_x = (screen_w / 2) + (offset_x * screen_w * SENSITIVITY_X)
        target_y = (screen_h / 2) + (offset_y * screen_h * SENSITIVITY_Y)


        curr_x = curr_x + (target_x - curr_x) * SMOOTHING
        curr_y = curr_y + (target_y - curr_y) * SMOOTHING

        pyautogui.moveTo(int(curr_x), int(curr_y))
        
      
        if cv2.waitKey(1) & 0xFF == ord('c'):
            calibrated = False

    cv2.imshow('Calibrated Eye Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()