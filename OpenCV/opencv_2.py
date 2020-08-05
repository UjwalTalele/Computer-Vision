# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 17:03:36 2020

@author: ujwal
"""

#importing packages
import argparse
import imutils 
import cv2

#construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required = True,
                help = "path to input image")
args = vars(ap.parse_args())

#load i/p image and display 
image = cv2.imread(args["image"])
cv2.imshow("Image", image)
cv2.waitKey(0)

#convert to greyscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)

#edge detection
edged = cv2.Canny(gray, 30,150)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

#thresholding the image for all pixel values to range of 225 to 250
thresh = cv2.threshold(gray,225,255,cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

#find contours(i.e. outline) of images input the threshold image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = image.copy()
#loop over contours
for c in cnts:
    cv2.drawContours(output,[c], -1,(240,0,159),3)
    cv2.imshow("Contours", output)
    cv2.waitKey(0)

#writing text on image
text = "I found {} objects".format(len(cnts))
cv2.putText(output, text, (10,25),cv2.FONT_HERSHEY_SIMPLEX, 0.7 ,
            (240,0,159),2)
cv2.imshow("Contours", output)
cv2.waitKey(0)


# we apply erosions to reduce the size of foreground objects
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=5)
cv2.imshow("Eroded", mask)
cv2.waitKey(0)

# similarly, dilations can increase the size of the ground objects
mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=5)
cv2.imshow("Dilated", mask)
cv2.waitKey(0)

# a typical operation we may want to apply is to take our mask and
# apply a bitwise AND to our input image, keeping only the masked
# regions
mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Output", output)
cv2.waitKey(0)


