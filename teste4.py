#import OpenCV library
import cv2
#import matplotlib library
#import matplotlib.pyplot as plt
#importing time library for speed comparisons of both classifiers
import time 
#%matplotlib inline

def convertToRGB(img): 
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
 #just making a copy of image passed, so that passed image is not changed 
    img_copy = colored_img.copy()          

 #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)          

 #let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);          

 #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        eyes_detected_img = detect_eyes(haar_eye_cascade, img_copy) 
    return eyes_detected_img

def detect_eyes(e_cascade, colored_img, scaleFactor = 1.13):
 #just making a copy of image passed, so that passed image is not changed 
    img_copy = colored_img.copy()          

 #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)          

 #let's detect multiscale (some images may be closer to camera than others) images
    eyes = e_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);          

 #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in eyes:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)              
    return img_copy


#load another image 
test2 = cv2.imread('data/test18.jpeg')
 

#load cascade classifier training file for haarcascade 
haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
haar_eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#haar_eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

#call our function to detect faces 
faces_detected_img = detect_faces(haar_face_cascade, test2)    


# or display the gray image using OpenCV 
cv2.imshow('Test Imag', convertToRGB(faces_detected_img)) 
cv2.waitKey(0) 
cv2.destroyAllWindows()
