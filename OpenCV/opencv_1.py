# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:57:16 2020

@author: ujwal
"""

#import libraries
import imutils 
import cv2

#load i/p image and show dimensions in form (width*height*depth)
image = cv2.imread("jp.png")
(h,w,d) = image.shape
print("width = {},height = {}, depth ={}".format(w,h,d))

#display image on screen
cv2.imshow("Image", image)
cv2.waitKey(0)

#accessing a bgr at certain location
(B,G,R) = image(100,50)
print("R={}, G={}, B={}".format(R, G, B))

#cropping with slicing 
# extract a 100x100 pixel square ROI (Region of Interest) from the
# input image starting at x=320,y=60 at ending at x=420,y=160
roi = image[60:160, 320:420]
cv2.imshow("ROI", roi)
cv2.waitKey(0)

#resizing images
resized = cv2.resize(image,(200,200))
cv2.imshow("Fixed Resizing", resized)
cv2.waitKey(0)
# this causes distortion

#fix by aaspect ratio
r = 300.0/w
dim = (300,int(h*r))
resized = cv2.resize(image,dim)
cv2.imshow("Aspect Ratio Resize: ", resized)
cv2.waitKey(0)

#resize using imutils
resized = imutils.resize(image, width = 300)
cv2.imshow("Imutils Resize: ", resized)
cv2.waitKey(0)

#rotating image from center by 45 degrees
center = (w//2,h//2)
M = cv2.getRotationMatrix2D(center,-45,1.0)
rotated = cv2.warpAffine(image, M, (w,h))
cv2.imshow("OpenCV Rotation: ", rotated)
cv2.waitKey(0)

#rotation using imutils
rotated = imutils.rotate(image,-45)
cv2.imshow("Imutils Rotated: ", rotated)
cv2.waitKey(0)

#rotate withut clipping
rotated = imutils.rotate_bound(image, -45)
cv2.imshow("Rotate vound Rotate: ", rotated)
cv2.waitKey(0)

#image smoothing using gaussian blur
blurred = cv2.GaussianBlur(image, (11,11),0)
cv2.imshow("Gaussian Blurred: ", blurred)
cv2.waitKey(0)

#insert objects
#2px thich red rectangle surrounding the face
output = image.copy()
cv2.rectangle(output, (200, 0), (429, 0), (0, 0, 255), 2)
cv2.imshow("Rectangle", output)
cv2.waitKey(0)

# draw a blue 20px (filled in) circle on the image centered at
# x=300,y=150
output = image.copy()
cv2.circle(output, (300, 150), 50, (255, 0, 0), -5)
cv2.imshow("Circle", output)
cv2.waitKey(0)

# draw a 5px thick red line from x=60,y=20 to x=400,y=200
output = image.copy()
cv2.line(output, (60, 20), (400, 200), (0, 0, 255), 5)
cv2.imshow("Line", output)
cv2.waitKey(0)

# draw green text on the image
output = image.copy()
cv2.putText(output, "OpenCV + Jurassic Park!!!", (10, 25), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Text", output)
cv2.waitKey(0)




