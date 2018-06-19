import numpy as np
import cv2
from matplotlib import pyplot as plt

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml') 
#olho_direito = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
#olho_esquerdo =  cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #blurred_image = cv2.GaussianBlur(gray,(5,5),0)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
       
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.35, 5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            roi_gray2 = gray[ey:ey+eh, ex:ex+ew]
            roi_color2 = img[ey:ey+eh, ex:ex+ew]
            retval, thresholded = cv2.threshold(roi_gray2, 30, 255, cv2.THRESH_BINARY)
            circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=55,param2=50,minRadius=0,maxRadius=0)
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                cv2.circle(cimg,(i[0],i[1]),i[2],(255,0,0),2)
                cv2.circle(cimg,(i[0],i[1]),2,(255,0,0),3)



    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
