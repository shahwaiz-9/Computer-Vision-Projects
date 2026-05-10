import cv2
import numpy as np
import pytesseract
from itertools import product



# GLOABALS

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# GAME SOLVING ALGOTITHM

def generate_placements(hints, length):
    """Generate all valid placements for a row/column given its hints."""
    results = []

    def recurse(h_idx, pos, current):
        if h_idx == len(hints):
            full = current + [-1] * (length - len(current))
            results.append(full)
            return
        h = hints[h_idx]
        remaining = sum(hints[h_idx+1:]) + len(hints[h_idx+1:])
        for p in range(pos, length - h - remaining + 1):
            line = current + [-1] * (p - len(current)) + [1] * h
            if h_idx < len(hints) - 1:
                line = line + [-1]
            recurse(h_idx + 1, len(line), line)

    recurse(0, 0, [])
    return results


def filter_placements(placements, known):
    """Keep only placements compatible with already-known cells."""
    return [p for p in placements if all(k == 0 or k == p[i] for i, k in enumerate(known))]


def overlap_cells(placements, length):
    """Find cells that are identical across ALL valid placements."""
    result = [0] * length
    for i in range(length):
        vals = [p[i] for p in placements]
        if all(v == 1 for v in vals):
            result[i] = 1
        elif all(v == -1 for v in vals):
            result[i] = -1
    return result


def solve(grid, row_hints, col_hints):
    N = len(grid)
    grid = [row[:] for row in grid]  # deep copy

    row_placements = [
        filter_placements(generate_placements(row_hints[r], N), grid[r])
        for r in range(N)
    ]
    col_placements = [
        filter_placements(generate_placements(col_hints[c], N), [grid[r][c] for r in range(N)])
        for c in range(N)
    ]

    if any(len(p) == 0 for p in row_placements + col_placements):
        return None

    # Constraint propagation loop
    changed = True
    while changed:
        changed = False

        for r in range(N):
            overlap = overlap_cells(row_placements[r], N)
            for c in range(N):
                if grid[r][c] == 0 and overlap[c] != 0:
                    grid[r][c] = overlap[c]
                    changed = True
            row_placements[r] = filter_placements(row_placements[r], grid[r])

        for c in range(N):
            col = [grid[r][c] for r in range(N)]
            overlap = overlap_cells(col_placements[c], N)
            for r in range(N):
                if grid[r][c] == 0 and overlap[r] != 0:
                    grid[r][c] = overlap[r]
                    changed = True
            col_placements[c] = filter_placements(col_placements[c], [grid[r][c] for r in range(N)])

        if any(len(p) == 0 for p in row_placements + col_placements):
            return None

    # Check if fully solved
    if all(grid[r][c] != 0 for r in range(N) for c in range(N)):
        return grid

    # Backtracking: find first unknown cell and try both values
    for r in range(N):
        for c in range(N):
            if grid[r][c] == 0:
                for val in [1, -1]:
                    new_grid = [row[:] for row in grid]
                    new_grid[r][c] = val
                    result = solve(new_grid, row_hints, col_hints)
                    if result:
                        return result
                return None

    return grid


def print_grid(grid):
    symbols = {1: "■", -1: "×", 0: "?"}
    for i, row in enumerate(grid):
        print(f"Row {i+1}: {row}   {''.join(symbols[v] for v in row)}")





# FUNCTIONS

def cropContours(contours, original_img):
    col_hints = []
    row_hints = []

    gaint_cnt = max(contours, key=cv2.contourArea)
    gx, gy, gw, gh = cv2.boundingRect(gaint_cnt)

    for cnt in contours:
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





def extract_values(hints, original_img):
    values = []
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    for i, cnt in enumerate(hints):
        x, y, w, h = cv2.boundingRect(cnt)
        roi = gray_img[y:y+h, x:x+w]
        
        # 1. Removing blue lines
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(thresh) < 127: # Ensure background is white
            thresh = cv2.bitwise_not(thresh)
            
        # 2. making numbers prominent by isolating the ink
        ink_only = cv2.bitwise_not(thresh)
        digit_cnts, _ = cv2.findContours(ink_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not digit_cnts:
            values.append("")
            continue

        # 3. isolating the bounding box of all ink and centering it on a blank canvas
        canvas_w, canvas_h = 100, 100
        canvas = np.ones((canvas_h, canvas_w), dtype="uint8") * 255
        
        # Get bounding box of all ink found in the box
        all_x, all_y, all_w, all_h = cv2.boundingRect(np.concatenate(digit_cnts))
        digit_roi = thresh[all_y:all_y+all_h, all_x:all_x+all_w]
        
        # 4. Center the digit(s) on the canvas
        offset_x = (canvas_w - all_w) // 2
        offset_y = (canvas_h - all_h) // 2
        canvas[offset_y:offset_y+all_h, offset_x:offset_x+all_w] = digit_roi


        config = '--psm 6 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(canvas, config=config).strip()
        
        clean_text = "".join(text.split())
        values.append(clean_text)

    return values



def format_hints(raw_hints):
    formatted = []
    for hint in raw_hints:
        if hint == '?' or hint == '':
            formatted.append([]) 
        else:
            # Split "21" into [2, 1]
            digits = [int(d) for d in hint]
            formatted.append(digits)
    return formatted



def get_warped_grid(img, grid_contour):
    # Simplify contour to get 4 corners
    peri = cv2.arcLength(grid_contour, True)
    approx = cv2.approxPolyDP(grid_contour, 0.02 * peri, True)
    
    if len(approx) == 4:
        # Sort corners: [top-left, top-right, bottom-right, bottom-left]
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        # Define destination (500x500 square)
        dst = np.array([[0, 0], [499, 0], [499, 499], [0, 499]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        invM = cv2.getPerspectiveTransform(dst, rect) # To warp back later
        return M, invM
    return None, None


def draw_solution(solution, invM, original_img):
    h, w = original_img.shape[:2]
    # 1. Create a flat black canvas (the Ghost Canvas)
    canvas = np.zeros((500, 500, 3), dtype="uint8")
    
    cell_size = 100
    for r in range(5):
        for c in range(5):
            # Calculate center of the cell for better placement
            center_x = c * cell_size + cell_size // 2
            center_y = r * cell_size + cell_size // 2
            
            if solution[r][c] == 1:
                # Solid Green Square
                cv2.rectangle(canvas, (c*cell_size + 10, r*cell_size + 10), 
                ((c+1)*cell_size - 10, (r+1)*cell_size - 10), (132, 17, 17), -1)

            elif solution[r][c] == -1:
                # Bright Red X
                cv2.line(canvas, (center_x - 30, center_y - 30), 
                         (center_x + 30, center_y + 30), (0, 0, 255), 5)
                cv2.line(canvas, (center_x + 30, center_y - 30), 
                         (center_x - 30, center_y + 30), (0, 0, 255), 5)

    # 2. Warp the drawing back to the photo's perspective
    warped_overlay = cv2.warpPerspective(canvas, invM, (w, h))
    
    # 3. Create a mask: anywhere that isn't black (0,0,0) in our warped drawing
    mask = cv2.cvtColor(warped_overlay, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    
    # 4. Copy the original image and 'stamp' the solution on top
    result = original_img.copy()
    result[mask > 0] = warped_overlay[mask > 0]
    
    return result







# MAIN 

img = cv2.imread('text.jpeg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)  
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  

print(f"Found {len(contours)} contours.")


col_hints, row_hints = cropContours(contours, img)
col_values = extract_values(col_hints, img)
row_values = extract_values(row_hints, img)



final_col_values = format_hints(col_values)
final_row_values = format_hints(row_values)



solution = solve([[0]*5 for _ in range(5)], final_row_values, final_col_values)

print("Solution:")
print_grid(solution)


print("Column Hints:", final_col_values)
print("Row Hints:", final_row_values)


M , invM = get_warped_grid(img, max(contours, key=cv2.contourArea))

# warped_view = cv2.warpPerspective(img, M, (500, 500))
# cv2.imshow("Debug: Warped Grid", warped_view)


print("Perspective Transform Matrix:", M)



print("Inverse Perspective Transform Matrix:", invM)
final_img = draw_solution(solution, invM, img)


cv2.imshow('Final Solution', final_img)
cv2.waitKey(0)

# for cnt in col_hints:
#     x, y, w, h = cv2.boundingRect(cnt)
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# for cnt in row_hints:
#     x, y, w, h = cv2.boundingRect(cnt)
#     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# cv2.imshow('Hints', img)
# cv2.waitKey(0)
