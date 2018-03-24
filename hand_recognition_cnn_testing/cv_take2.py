# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

count = 0
active = False
while True:
    #time.sleep(0.1)
    frame = cap.read()[1]
    
    
    #lower = np.array([100, 48, 80], dtype = "uint8")
    #upper = np.array([200, 255, 255], dtype = "uint8")
    #mask = cv2.inRange(frame,lower,upper)
    #if it's in the mask range itll be one else 0
    #frame = cv2.bitwise_and(frame,frame,mask=mask)

    
    #coz initial recording orientation was sideways and autistic
    #frame =  cv2.transpose(frame).
    #frame = cv2.flip(frame,0)
    
    #crop
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #Apply skin color range
    mask = cv2.inRange(hsv, low_range, upper_range)

    mask = cv2.erode(mask, skinkernel, iterations = 1)
    mask = cv2.dilate(mask, skinkernel, iterations = 1)
    
    #blur
    mask = cv2.GaussianBlur(mask, (15,15), 1)
    #cv2.imshow("Blur", mask)
    
    #bitwise and mask original frame
    res = cv2.bitwise_and(roi, roi, mask = mask)
    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    
    
    frame = frame[:480,:480]
    
    
    cv2.imshow('original',res)
    #frame = cv2.resize(frame,(128,128))
    if cv2.waitKey(1) & 0xFF == ord('f'):
        active = False if active else True
    if active:
        cv2.imwrite("fresh_data/five"+str(count)+".jpg",frame)
        print("cap count:",count)
        count+=1
    if cv2.waitKey(2) & 0xFF == ord(' '):
        break
cap.release()
cv2.destroyAllWindows()
