import cv2
import numpy as np
import easyocr
import os

# --- INITIALIZATION ---
# Keep your Deep Learning engine ready
reader = easyocr.Reader(['en'], gpu=False)

# Load Template Library
digit_templates = {}
template_path = 'templates/'
if os.path.exists(template_path):
    for i in range(1, 5):
        t_img = cv2.imread(f'{template_path}{i}.jpeg', 0)
        if t_img is not None:
            digit_templates[str(i)] = t_img

# --- CORE FUNCTIONS ---

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    return thresh

def extract_countors(countors):
    col_hints = []
    row_hints = []
    gaint_cnt = max(countors, key=cv2.contourArea)
    gx, gy, gw, gh = cv2.boundingRect(gaint_cnt)

    for cnt in countors:
        curr_x, curr_y, curr_w, curr_h = cv2.boundingRect(cnt)
        if curr_w > gw * 0.8: continue 
        
        center_x = curr_x + curr_w // 2
        center_y = curr_y + curr_h // 2
        
        if center_y < (gy - 5): 
            col_hints.append(cnt)
        elif center_x < (gx - 5): 
            row_hints.append(cnt)

    col_hints.sort(key=lambda c: cv2.boundingRect(c)[0]) 
    row_hints.sort(key=lambda c: cv2.boundingRect(c)[1]) 
    return col_hints, row_hints

# --- ENGINE 1: DEEP LEARNING (EasyOCR) ---
def get_values_easyocr(hint_roi):
    results = reader.readtext(hint_roi)
    results.sort(key=lambda res: res[0][0][1]) # Sort by Y
    box_values = []
    for (bbox, text, prob) in results:
        clean_text = "".join([c for c in text if c.isdigit()])
        if clean_text:
            box_values.append(clean_text)
    return box_values

# --- ENGINE 2: TEMPLATE MATCHING ---
def get_values_template(hint_roi, templates):
    found_digits = []
    if len(hint_roi.shape) == 3:
        hint_roi = cv2.cvtColor(hint_roi, cv2.COLOR_BGR2GRAY)

    for val, temp_img in templates.items():
        res = cv2.matchTemplate(hint_roi, temp_img, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= 0.8) # 80% match threshold
        for pt in zip(*loc[::-1]):
            found_digits.append((pt[1], val)) # (Y-coord, digit)
            break # Only need one instance of each unique digit per box
            
    found_digits.sort(key=lambda x: x[0])
    return [d[1] for d in found_digits]

# --- MAIN WRAPPER ---
def get_all_hint_data(hints, original_img, mode='easyocr'):
    all_values = []
    for i, cnt in enumerate(hints):
        x, y, w, h = cv2.boundingRect(cnt)
        hint_roi = original_img[y:y+h, x:x+w]
        
        if mode == 'template' and digit_templates:
            detected = get_values_template(hint_roi, digit_templates)
        else:
            detected = get_values_easyocr(hint_roi)
            
        all_values.append(detected)
        # print(f"Box {i}: {detected}")
    return all_values

# --- EXECUTION ---
img = cv2.imread('text.jpeg')
if img is None:
    print("Error: 'text.jpeg' not found.")
else:
    thresh = preprocess(img)
    counters = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    col_hints, row_hints = extract_countors(counters)

    # CHOOSE YOUR ENGINE: 'easyocr' or 'template'
    engine_mode = 'easyocr' 
    
    print(f"Running extraction using: {engine_mode}")
    col_values = get_all_hint_data(col_hints, img, mode=engine_mode)
    row_values = get_all_hint_data(row_hints, img, mode=engine_mode)

    print("\n--- RESULTS ---")
    print("Columns:", col_values)
    print("Rows:   ", row_values)

    # To Save templates for the first time, uncomment below:
    # for i, cnt in enumerate(col_hints):
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     cv2.imwrite(f'debug_hint_{i}.png', img[y:y+h, x:x+w])

    cv2.imshow('Final Grid Detection', cv2.resize(thresh, (0,0), fx=0.5, fy=0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()