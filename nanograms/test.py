import cv2
import numpy as np
import easyocr

# Initialize Reader once at the start
reader = easyocr.Reader(['en'], gpu=False)

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Using a slightly larger block size (15) to keep number shapes cleaner
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    return thresh

def extract_countors(countors):
    col_hints = []
    row_hints = []

    gaint_cnt = max(countors, key=cv2.contourArea)
    gx, gy, gw, gh = cv2.boundingRect(gaint_cnt)

    for cnt in countors:
        curr_x, curr_y, curr_w, curr_h = cv2.boundingRect(cnt)
        
        # Avoid the grid AND anything inside the grid
        if curr_w > gw * 0.8: continue 
        
        center_x = curr_x + curr_w // 2
        center_y = curr_y + curr_h // 2
        
        # Use a 5-pixel buffer to ensure we don't grab grid-line artifacts
        if center_y < (gy - 5): 
            col_hints.append(cnt)
        elif center_x < (gx - 5): 
            row_hints.append(cnt)

    col_hints.sort(key=lambda c: cv2.boundingRect(c)[0]) 
    row_hints.sort(key=lambda c: cv2.boundingRect(c)[1]) 

    return col_hints, row_hints

def get__values_easyocr(hints, original_img):
    values = []
    for i, cnt in enumerate(hints):
        x, y, w, h = cv2.boundingRect(cnt)
        # Crop from ORIGINAL image (EasyOCR likes color/gray over thresh)
        # Add 2px padding to help the AI see the edges
        hint_roi = original_img[max(0, y-2):y+h+2, max(0, x-2):x+w+2] 

        results = reader.readtext(hint_roi)
        
        # Sort vertically for stacked numbers
        results.sort(key=lambda res: res[0][0][1]) 

        box_values = []
        for (bbox, text, prob) in results:
            clean_text = "".join([c for c in text if c.isdigit()])
            if clean_text:
                box_values.append(clean_text)
        
        values.append(box_values)
    return values

# --- Main Execution ---
img = cv2.imread('text.jpeg')
if img is None:
    print("Error: Could not find 'text.jpeg'")
else:
    thresh = preprocess(img)
    counters = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    col_hints, row_hints = extract_countors(counters)

    # Use 'img' for OCR, not 'thresh'
    col_values = get__values_easyocr(col_hints, img)
    row_values = get__values_easyocr(row_hints, img)

    print("\n--- FINAL RESULTS ---")
    print("Column Hints:", col_values)
    print("Row Hints:   ", row_values)

    # Visualization
    resized = cv2.resize(thresh, (0,0), fx=0.5, fy=0.5)
    cv2.imshow('Detection Process', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()