import numpy as np
import cv2

# Logical Questions 
# 1. How cv2 convert the color image to grayscale?
# Does gaussian blur is necessary for all the images?
# yes it makes image clearer and smooth and it is good to have this in first place
# 

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale
    # blur = cv2.GaussianBlur(gray, (5, 5), 0) # apply Gaussian blur to reduce noise
    blur = cv2.bilateralFilter(gray, 9, 75, 75)

    v = np.median(blur)

    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(blur, lower, upper)

    # th = cv2.adaptiveThreshold(
    #     blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 2
    # )

    return edges

def ind_board(thresholded):
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000: # filter out small contours
            x, y, w, h = cv2.boundingRect(contour)
            return (x, y, w, h) # return the bounding box of the detected board
        
    return None


def find_board(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    return None

cap = cv2.VideoCapture(0) # 0 is the default camera

while True:

    ret , frame = cap.read() # read a frame from the camera


    if not ret:
        print("Failed to grab frame")
        break


    threshold = preprocess_frame(frame) # preprocess the frame (e.g., convert to grayscale)
    board = find_board(threshold) # identify the board in the thresholded image

    if board is not None:
        cv2.drawContours(frame, [board], -1, (0, 255, 0), 2) # draw the detected board contour on the original frame

    cv2.imshow("Camera Feed", frame)
    cv2.imshow("Edges", threshold)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




# import cv2
# import numpy as np

# def preprocess(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     th = cv2.adaptiveThreshold(
#         blur, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV,
#         11, 2
#     )
#     return th

# def order_points(pts):
#     pts = pts.reshape(4, 2)
#     s = pts.sum(axis=1)
#     diff = np.diff(pts, axis=1)

#     tl = pts[np.argmin(s)]
#     br = pts[np.argmax(s)]
#     tr = pts[np.argmin(diff)]
#     bl = pts[np.argmax(diff)]
#     return np.array([tl, tr, br, bl], dtype=np.float32)

# def find_board(thresh):
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)

#     for c in contours[:10]:
#         peri = cv2.arcLength(c, True)
#         approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#         if len(approx) == 4:
#             return approx
#     return None

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     thresh = preprocess(frame)
#     board = find_board(thresh)

#     display = frame.copy()

#     if board is not None:
#         cv2.drawContours(display, [board], -1, (0, 255, 0), 2)

#         pts = order_points(board)
#         size = 500
#         dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)

#         M = cv2.getPerspectiveTransform(pts, dst)
#         warped = cv2.warpPerspective(frame, M, (size, size))

#         cv2.imshow("Warped", warped)

#     cv2.imshow("Frame", display)
#     cv2.imshow("Thresh", thresh)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




