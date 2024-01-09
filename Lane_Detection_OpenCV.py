import cv2
import numpy as np


def draw_lines(img, lines):
    lines_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8) # another image for the lines

    # x,y are starting and ending points of the lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(lines_image, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)

    image_with_lines = cv2.addWeighted(img, 1, lines_image, 1, 0.0) # merge the frame with the lines
    return image_with_lines


def ROI(img, ROI_pionts):
    mask = np.zeros_like(img) # an array of zeros (black) with the same size as the frame
    cv2.fillPoly(mask, ROI_pionts, 255) # white triangle ROI
    masked_img = cv2.bitwise_and(img, mask) # replace white white triangle ROI with the original frame
    return masked_img


def detected_lanes(img):
    height = img.shape[0]
    width = img.shape[1]

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny_img = cv2.Canny(gray_img, 100, 120) # edge detection

    ROI_pionts = [(0, height), (width / 2, height * 0.65), (width, height)] # triangle ROI (bottom-left corner, middle, bottom-right corner)
    
    crop_img = ROI(canny_img, np.array([ROI_pionts], np.int32)) # crop triangle ROI

    # line detection algorithm
    lines = cv2.HoughLinesP(crop_img, rho=2, theta=np.pi / 180, threshold=50, lines=np.array([]), minLineLength=40, maxLineGap=150)

    image_lines = draw_lines(img, lines)

    return image_lines



#--------------
cap = cv2.VideoCapture('lane_detection_video.mp4')

while (cap.isOpened()):
    ret, frame = cap.read()

    if ret ==True:
        frame = detected_lanes(frame)

        cv2.imshow('Lane Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
