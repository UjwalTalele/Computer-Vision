# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 21:06:49 2020

@author: ujwal
"""

from transformer.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

#construct arguments and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required = True,
                help = "Path to image")
args = vars(ap.parse_args())

##Edge detection

#load and compute ratio of old height to new , clone and resize
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image,height = 500)

#convert to greyscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(gray, 75,200)

#show original and edge detected
print("Step 1: Edge detection")
cv2.imshow("Image", image)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Finding Contours
#find contours in edged image keeping largest ones and intitialize screen counter
cnts = cv2.findContours(edges.copy(),cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts,key = cv2.contourArea, reverse = True)[:5]

#loop over contours
for c in cnts:
    #approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 *peri,True)
    
    #if approximated contour has four points then we assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break

#show the contours
print("Step 2: Find contours of paper")
cv2.drawContours(image,[screenCnt], -1, (0,255,0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Apply perspective transform and threshold

#apply four point transform
warped = four_point_transform(orig, screenCnt.reshape(4,2) * ratio)

#convert warped image to grayscale
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

#show the actual and scanned image
print("Step 3: Apply prespective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height= 650))
cv2.waitKey(0)

