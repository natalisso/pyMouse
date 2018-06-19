import cv2
import numpy as np
from matplotlib import pyplot as plt

bgr_img = cv2.imread('img/sid.jpeg') 

if bgr_img.shape[-1] == 3:           
    b,g,r = cv2.split(bgr_img)      
    rgb_img = cv2.merge([r,g,b])     
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
else:
    gray_img = bgr_img

cimg = gray_img.copy()

#cimg = cv2.cvtColor(bgr_img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(gray_img,cv2.HOUGH_GRADIENT,1,20,
                            param1=55,param2=50,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))
print(circles[0,:][0][0])
for i in circles[0,:]:
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

plt.subplot(121),plt.imshow(rgb_img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(cimg)
plt.title('Hough Transform'), plt.xticks([]), plt.yticks([])
plt.show()
print(rgb_img.shape)
b = rgb_img[int(circles[0,:][0][0])-100:int(circles[0][:][0])+100, int(circles[0][:][0][1])-100:int(circles[0][:][0][1])+100,:]
plt.imshow(b)
plt.show()
