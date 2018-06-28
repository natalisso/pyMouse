import cv2
import numpy as np
import pyautogui
from matplotlib import pyplot as plt

template = cv2.imread('img/webcam4.jpg',0)
w1, h1 = template.shape[::-1]
method = eval('cv2.TM_CCORR_NORMED')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

# Exit if video not opened.
if not cap.isOpened():
    print ("Could not open video")
    sys.exit()
    
# Read first frame.
ok, frame = cap.read()
if not ok:
    print ('Cannot read video file')
    sys.exit()
    
#409 - 23 (x)
#512 - 30 (y)

count = 0
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi_gray = gray.copy()
    #blurred_image = cv2.GaussianBlur(gray,(5,5),0)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.35, 5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            roi_gray2 = roi_gray[ey:ey+eh, ex:ex+ew]
            roi_color2 = roi_color[ey:ey+eh, ex:ex+ew]
            res = cv2.matchTemplate(roi_gray2,template,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            top_left = max_loc
            bottom_right = (top_left[0] + w1, top_left[1] + h1)
            #cv2.rectangle(roi_color2,top_left, bottom_right, 255, 2)
            cv2.circle(roi_color2,(int(top_left[0]+w1/2),int(top_left[1]+h1/2)), 4, (0,0,255), -1)
            #pyautogui.moveTo(int(top_left[0]+w1/2),int(top_left[1]+h1/2), duration = 1)
            if count == 20:
                pyautogui.moveTo(int((top_left[0]+w1/2)*409/23),int((top_left[1]+h1/2)*512/30), duration = 0.005)
                count = 0
    count+=1



    cv2.imshow('img',img)
    #cv2.imshow('gray',gray)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
