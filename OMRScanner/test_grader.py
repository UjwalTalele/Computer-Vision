# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 14:37:16 2020

@author: ujwal
"""

from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse 
import imutils 
import cv2


#construct arguments and parse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
                help = "path to input image")
args = vars(ap.parse_args())

#define answer key which maps question no. to correct answer
ANSWER_KEY = {0:1, 1:4, 2:0, 3:3, 4:1}

#load image, convert to grayscale blur and find edges
image = cv2.imread(args["image"])
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred =cv2.GaussianBlur(gray,(5,5), 0) 
edged = cv2.Canny(blurred, 75, 200)

#find contours in edge and then initialize the contour that 
# corresponds to document
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

##ensure at least one contour was found
if len(cnts) > 0:
    #sort cnts according to size in descending order
    cnts = sorted(cnts, key= cv2.contourArea, reverse = True)
    
# loop over sorted contours
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    # if approximate contour has four points then we assume that we have found the paper
    if len(approx) == 4:
        docCnt = approx
        break


#apply four point perspective transform on original as welll as grayscale image to obtain top-down birds eye view
paper = four_point_transform(image, docCnt.reshape(4,2))
warped = four_point_transform(gray, docCnt.reshape(4,2))

#apply Otsu's thresholding to binarize the wrapped paper
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

## now applying contour extracing to each of the circle

# find contours in threshold image, then initialize list of contours that correspond to questions
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

#loop over contours
for c in cnts:
    #conpute bounding box of contour then use bounding box to derive aspect ratio
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w/float(h)
    
    #to label contour as question, region should be suffeciently wide,tall
    # and should have aspect ratio approximately equal to 1
    if w >= 20 and h>=20 and ar >= 0.9 and ar<=1.1:
        questionCnts.append(c)
        
#GRADING PART

#sort question contours top_to_bottom then initialize total number of correct answer
questionCnts = contours.sort_contours(questionCnts, 
                                      method = "top-to-bottom")[0]
correct= 0

# each question has 5 possible answers, to loop over question in batches of 5

for (q,i) in enumerate(np.arange(0, len(questionCnts), 5)):
    #sort contours for current question from left to right 
    # then initialize index of bubbled answer
    cnts = contours.sort_contours(questionCnts[i:i+5])[0]
    bubbled = None
    
    #loop over sorted contours
    for (j,c) in enumerate(cnts):
        #construct  a mask that only reveals only current "bobble" for question
        mask = np.zeros(thresh.shape , dtype = "uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        #apply the mask to the threshold image, then count non zero pixels in bubble area
        mask = cv2.bitwise_and(thresh, thresh, mask = mask)
        total = cv2.countNonZero(mask)
        
        #if current total has a larger number of total non-zero pixels then
        # we are examining the currently bubbled in answer
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)
    
    
    # initialize contour color and index of correct answer
    color = (0,0,255)
    k = ANSWER_KEY[q]
    
    #check if bubbled answer is correct
    if k == bubbled[1]:
        color = (0,255,0)
        correct += 1
    
    # draw outline of corrrect answer on text
    cv2.drawContours(paper, [cnts[k]], -1, color, 3)

# grab the test taker
score = (correct / 5.0) * 100
#print("{INFO} score :{:.2f}%".format(score))
cv2.putText(paper, "{:.2f}".format(score), (10,30), 
            cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", paper)
cv2.waitKey(0)
         
    



