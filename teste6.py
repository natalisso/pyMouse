import cv2
import numpy as np
from matplotlib import pyplot as plt

bgr_img = cv2.imread('img/eye.jpg') 

if bgr_img.shape[-1] == 3:           
    b,g,r = cv2.split(bgr_img)   
    rgb_img = cv2.merge([r,g,b])     
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
else:
    gray_img = bgr_img

img = cv2.medianBlur(gray_img, 15)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
retval, thresholded = cv2.threshold(cimg, 30, 255, cv2.THRESH_BINARY)

circles1 = cv2.HoughCircles(thresholded,cv2.HOUGH_GRADIENT,1,20,
                            param1=55,param2=50,minRadius=0,maxRadius=0)

circles1 = np.uint16(np.around(circles1))

for i in circles1[0,:]:
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

plt.subplot(121),plt.imshow(thresholded)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(cimg)
plt.title('Hough Transform'), plt.xticks([]), plt.yticks([])
plt.show()
