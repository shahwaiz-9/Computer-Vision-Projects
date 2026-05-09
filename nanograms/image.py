import cv2
import numpy as np
import pytesseract
import easyocr

reader = easyocr.Reader(['en'], gpu=False)

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh


def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def get_contours(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(f"Found {len(contours)} contours")
    # print(f"Contours: {contours}")
    return contours







# def visualize_contours(image, contours, thresh):
#     for i, cnt in enumerate(counters):
    
#     mask = np.zeros_like(thresh)
    
#     cv2.drawContours(mask, [cnt], -1, 255, -1)
    

#     x, y, w, h = cv2.boundingRect(cnt)

#     display_mask = cv2.resize(mask, (int(mask.shape[1]*0.5), int(mask.shape[0]*0.5)))
    
#     print(f"Displaying Contour #{i} - Position: ({x}, {y}), Width: {w}, Height: {h}")
#     cv2.imshow(f'Contour {i}', display_mask)
    
#     print("Press any key to see the next shape...")
#     cv2.waitKey(0)
#     cv2.destroyWindow(f'Contour {i}')






# def order_points(pts):
#     pts = pts.reshape(4, 2)
#     s = pts.sum(axis=1)
#     diff = np.diff(pts, axis=1)

#     tl = pts[np.argmin(s)]
#     br = pts[np.argmax(s)]
#     tr = pts[np.argmin(diff)]
#     bl = pts[np.argmax(diff)]
#     return np.array([tl, tr, br, bl], dtype=np.float32)


def crop_grid(img, counters):

    
    gaint_cnt = max(counters, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(gaint_cnt)
    cropped = img[y:y+h, x:x+w]
    return cropped




# def extract_countors(countors):
    
#     col_hints = []
#     row_hints = []

#     gaint_cnt = max(countors, key=cv2.contourArea)
#     gx, gy, gw, gh = cv2.boundingRect(gaint_cnt)


#     for cnt in countors:

#         # if cnt == gaint_cnt:
#         #     continue


#     x, y, w, h = cv2.boundingRect(cnt)
#     center_x = x + w//2
#     center_y = y + h//2
    
#     # Classify contours based on their position relative to the grid 
#     # following nanogram structure where hints are placed above and to the left of the grid

#     if center_y < gy: # Above the grid
#         col_hints.append(cnt)
#     elif center_x < gx: # Left of the grid
#         row_hints.append(cnt)


#     return col_hints, row_hints


def extract_countors(countors):
    col_hints = []
    row_hints = []

   
    gaint_cnt = max(countors, key=cv2.contourArea)
    gx, gy, gw, gh = cv2.boundingRect(gaint_cnt)

    for cnt in countors:
        
        curr_x, curr_y, curr_w, curr_h = cv2.boundingRect(cnt)
        if curr_x == gx and curr_y == gy:
            continue

        
        center_x = curr_x + curr_w // 2
        center_y = curr_y + curr_h // 2
        
        
        if center_y < gy: # It is above the grid
            col_hints.append(cnt)
        elif center_x < gx: # It is to the left of the grid
            row_hints.append(cnt)


    col_hints.sort(key=lambda c: cv2.boundingRect(c)[0]) # Sort column hints by their x-coordinate (left to right)   
    row_hints.sort(key=lambda c: cv2.boundingRect(c)[1]) # Sort row hints by their y-coordinate (top to bottom)

    return col_hints, row_hints



# def get__values(hints, dimensions):
#     values = []

#     for i, cnt in enumerate(hints):
#         x, y, w, h = cv2.boundingRect(cnt)
#         # 1. Initial crop from threshold
#         hint_img = thresh[y:y+h, x:x+w]
        
#         # --- CLEANING PHASE (Visualize this!) ---
#         # Find contours inside this small hint box
#         inner_cnts, _ = cv2.findContours(hint_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         # Create a fresh white canvas (200x200) for a clean OCR environment
#         canvas = np.ones((200, 200), dtype=np.uint8) * 255
        
#         digit_list = []
#         for ic in inner_cnts:
#             ix, iy, iw, ih = cv2.boundingRect(ic)
            
#             # FILTER: If it's too big, it's the box. If it's too small, it's noise.
#             if ih > h * 0.8 or iw > w * 0.8:
#                 continue
#             if ih < 5: 
#                 continue
                
#             # Extract only the digit pixels
#             digit_pix = hint_img[iy:iy+ih, ix:ix+iw]
#             digit_list.append((iy, digit_pix))

#         # Sort by Y so stacked numbers (like 2 over 1) are in the right order
#         digit_list.sort(key=lambda x: x[0])

#         # Paste digits onto the white canvas
#         cursor_y = 20
#         for _, pix in digit_list:
#             # Resize 4x larger and thicken (dilate) to help Tesseract
#             pix_large = cv2.resize(pix, (0,0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
#             kernel = np.ones((3,3), np.uint8)
#             pix_bold = cv2.dilate(pix_large, kernel, iterations=1)
            
#             h_l, w_l = pix_bold.shape
#             # Paste onto canvas (Inverted to Black-on-White)
#             canvas[cursor_y:cursor_y+h_l, 40:40+w_l] = cv2.bitwise_not(pix_bold)
#             cursor_y += h_l + 10

#         # --- VISUALIZATION ---
#         cv2.imshow('1. Raw Hint', hint_img)
#         cv2.imshow('2. Cleaned for OCR', canvas)
#         cv2.waitKey(0) 
#         cv2.destroyAllWindows()

#         # --- OCR PHASE ---
#         # PSM 6 is better for "blocks" of text (helpful for stacked numbers)
#         custom_config = r'--psm 6 -c tessedit_char_whitelist=0123456789'
#         text = pytesseract.image_to_string(canvas, config=custom_config).strip()
        
#         # Log and store
#         print(f"Hint {i} at ({x},{y}): Found digits {text.split()}")
#         values.append(text.split()) # Store as a list to handle stacked numbers

#     print(f"Final Extracted Matrix: {values}")
#     return values



def get__values_easyocr(hints, dimensions):
    values = []

    for i, cnt in enumerate(hints):
        x, y, w, h = cv2.boundingRect(cnt)
        # Use a slightly padded crop from the original image or grayscale
        # EasyOCR prefers color or grayscale over harsh binary thresh
        hint_roi = img[y:y+h, x:x+w] 

        # readtext returns: [([[x,y], [x,y], ...], 'text', confidence), ...]
        results = reader.readtext(hint_roi)

        # We need to handle the "stacked" numbers (2 over 1)
        # Sort results by the y-coordinate of their bounding box
        results.sort(key=lambda res: res[0][0][1]) 

        box_values = []
        for (bbox, text, prob) in results:
            # Clean the text (EasyOCR sometimes picks up symbols)
            clean_text = "".join([c for c in text if c.isdigit()])
            if clean_text:
                box_values.append(clean_text)
                print(f"Hint {i}: Found '{clean_text}' with {prob:.2f} confidence")

        values.append(box_values)
    
    return values    

# main header


img = cv2.imread('text.jpeg')
thresh = preprocess(img)
resized= resize_image(thresh, 50)
counters = get_contours(thresh)


max_contour = max(counters, key=cv2.contourArea)
print(f"Max Contour Area: {cv2.contourArea(max_contour)}, Points: {max_contour}")


# Extract column and row hints based on their position relative to the largest contour (the grid)
col_hints , row_hints = extract_countors(counters)

# Extrating information about the hints for debugging purposes
col_values = get__values_easyocr(col_hints, "Columns")
row_values = get__values_easyocr(row_hints, "Rows")

print("Column Hints Values:", col_values)
print("Row Hints Values:", row_values)


print(f"Found {len(col_hints)} column hints and {len(row_hints)} row hints.")


# cropped = crop_grid(img, counters)
# resized_cropped = resize_image(cropped, 50)





# visualize_contours(img, counters, thresh)

# cv2.drawContours(resized, [max_contour], -1, (0, 255, 0), 2)
# cv2.imshow('Cropped Image', resized_cropped)
# cv2.imshow('Contours', resized)
cv2.imshow('Resized Image', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()