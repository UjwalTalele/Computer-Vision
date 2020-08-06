# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 19:13:32 2020

@author: ujwal
"""

import transform 
import numpy as np
import argparse 
import cv2

#construct argument parse and arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", 
                help = "path to image")
ap.add_argument("-c","--coords",
                help = "comma seperated list of source points")
args=  vars(ap.parse_args())

#load image and grab source co-ordinates 

image = cv2.imread(args["image"])
pts = np.array(eval(args["coords"]), dtype = "float32")

#apply four point transform to obtain birds eye view
warped = transform.four_point_transform(image,pts)

#show original and warped image
cv2.imshow("Original", image)
cv2.imshow("warped", warped)
cv2.waitKey(0)