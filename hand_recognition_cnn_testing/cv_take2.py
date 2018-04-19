# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#29 15 40
#145 130 158

import cv2
import numpy as np
import time
import pyautogui
import test1
from win32api import keybd_event



def press(Key, speed=1):
    rest_time = 0.05/speed
    #keydown
    keybd_event(Key, 0, 1, 0)
    time.sleep(rest_time)
    #keyup
    keybd_event(Key, 0, 2, 0)
 

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
frame = cap.read()[1]
cv2.namedWindow('bars')

cv2.createTrackbar('lH','bars',0,255,nothing)
cv2.createTrackbar('lS','bars',0,255,nothing)
cv2.createTrackbar('lV','bars',0,255,nothing)

cv2.createTrackbar('uH','bars',0,255,nothing)
cv2.createTrackbar('uS','bars',0,255,nothing)
cv2.createTrackbar('uV','bars',0,255,nothing)

cv2.setTrackbarPos('lH','bars',0)
cv2.setTrackbarPos('lS','bars',0)
cv2.setTrackbarPos('lV','bars',0)
cv2.setTrackbarPos('uH','bars',255)
cv2.setTrackbarPos('uS','bars',255)
cv2.setTrackbarPos('uV','bars',255)


count = 0
active = False
bg = None
trimbg = False
mouse_slow = False
new_center = None
pyautogui.FAILSAFE = False
screen_cam_ratioxy = None
brains = True
keyT_or_mouseF = False
         
d_thresh = 30 if(mouse_slow) else 3    
m_thresh = 0.3 if(mouse_slow) else 2.2

inference = None
lastI = None

if(brains):
    test1.load_weights()
    font = cv2.FONT_HERSHEY_SIMPLEX


while True:
    #time.sleep(0.1)
    frame = cap.read()[1]
    frame = cv2.flip(frame,1)

    lH = cv2.getTrackbarPos('lH','bars')
    lS = cv2.getTrackbarPos('lS','bars')
    lV = cv2.getTrackbarPos('lV','bars')
    
    uH = cv2.getTrackbarPos('uH','bars')
    uS = cv2.getTrackbarPos('uS','bars')
    uV = cv2.getTrackbarPos('uV','bars')
    
    lower = np.array([lH, lS, lV], dtype = "uint8")
    upper = np.array([uH, uS, uV], dtype = "uint8")
    
   
    mask = cv2.inRange(frame,lower,upper)
    #if it's in the mask range itll be one else 0
    frame = cv2.bitwise_and(frame,frame,mask=mask)

    
    #coz initial recording orientation was sideways and autistic
    #frame =  cv2.transpose(frame).
    
    
    #smoothing/averaging
    #divide by 15x15=225,(if (5,5) then 25)
    #kernel = np.ones((15,15),np.float32)/225
    #frame= cv2.filter2D(frame,-1,kernel)
    
    

    #blurring
    #blur = cv2.GaussianBlur(frame,(15,15),0)

    #erosion and dilation(expanding,squeezing pixel distributions)
    #kernel = np.ones((5,5),np.uint8)
    #erosion = cv2.erode(frame, kernel, iterations=1)
    #frame = cv2.dilate(erosion, kernel, iterations=1)


    #opening/ remove false positives, closing/remove false negatives
    #opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN,kernel)
    #frame = cv2.morphologyEx(opening, cv2.MORPH_CLOSE,kernel)
        
    #256x256 crop
    x = frame[:256,384:]


    if(trimbg):
        x = cv2.absdiff(bg,x)
        
        mask = cv2.inRange(x,lower,upper)
        #if it's in the mask range itll be one else 0
        x = cv2.bitwise_and(x,x,mask=mask)

        
        # color to grayscale
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    
        ret2, x= cv2.threshold(x,20,255,cv2.THRESH_BINARY)
        #256x256 crop
        #x = frame[:256,384:]
        #top, left, right, bottom
        x[:2,:] = x[:,:2] = x[:,-2:] =  x[-2:,:] = 0
        
        if(brains):
            inference = test1.get(cv2.resize(x,(64,64)).reshape((64,64,1))/255)
            cv2.putText(frame,inference,(0,50),font,2,(255,255,255),2,cv2.LINE_AA)
        if(keyT_or_mouseF):
            if(inference=='index'):
                #right
                press(32)  
            '''
            elif(inference=='index'):
                #right
                press(32)
            elif(inference=='index'):
                #right
                press(32)
            elif(inference=='index'):
                #right
                press(32)
            elif(inference=='index'):
                #right
                press(32)
            '''
            
        if(not keyT_or_mouseF):
            i, contours, heirarchy = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
            cont_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:5] if (not keyT_or_mouseF) else None
            exists = False
            
            #remember x row count is the height and column count is the width 
            
            if(cont_sorted):
                for i in range(len(cont_sorted)):
                    contrec = cv2.boundingRect(cont_sorted[i])
                    if 50 < contrec[2] < 250:
                        x1,y1,w,h = contrec
                        h = w
                        cv2.rectangle(x,(x1,y1),(x1+w,y1+ h ),(255,255,255),1)
                        exists = True
                        break
                center = (x.shape[1]//2,int(x.shape[0]/2*1.2))
                cv2.circle(x,center,4,[255,255,255],2)
                cv2.circle(x,center,1,[0,0,0],2)
                if(exists):
                        if (not new_center):
                            new_center = center
                        #screenwidth/camwidth, scheight/camheight
                        if(not screen_cam_ratioxy):
                            screen_cam_ratioxy = (pyautogui.size()[0]/x.shape[1],pyautogui.size()[1]/x.shape[0])
                            
                        hcenter = (x1+(w//2),y1+(h//2))
                        cv2.circle(x,hcenter,2,[0,0,0],1)
                        # sqrt( (x2-x1)^2 + (y2-y1)^2 )
                        d = np.sqrt( (new_center[0]-hcenter[0])**2 + (new_center[1]-hcenter[1])**2 )  
                        #print("distance: ",d)
                        if(d>d_thresh and inference=='Fist'):
                            
                            if(d<5):
                                m_thresh = 0.5
                            elif(d<12):
                                m_thresh = 1
                            elif(d<15):
                                m_thresh = 2
                            elif(d<18):
                                m_thresh = 3
                            elif(d>18):
                                m_thresh = 4
                            print(d)
                            
                            
                            nx = int((hcenter[0]-new_center[0])*screen_cam_ratioxy[0] * m_thresh)
                            ny = int((hcenter[1]-new_center[1])*screen_cam_ratioxy[1] * m_thresh)
                            pyautogui.moveRel(nx,ny)
                            if(not mouse_slow):
                                new_center = hcenter
                        elif(inference=='index' and lastI!='index'):
                            pyautogui.click()
                        elif(inference=='peace' and lastI!='peace'):
                            pyautogui.doubleClick()
            lastI = inference

    else:
        x = frame[:256,384:]
    
    y = x
    
    cv2.rectangle(frame, (639,0),(384,256),(255,255,255),2)
    #x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x,(64,64))


    cv2.imshow('image',frame)
    cv2.imshow('x',x)
    
    if cv2.waitKey(1) & 0xFF == ord('b'):
        if(trimbg):
            trimbg = False
        else:
            bg = y
            print("bg cap")
            trimbg = True
    
    if cv2.waitKey(1) & 0xFF == ord('m'):
        keyT_or_mouseF = False if(keyT_or_mouseF) else True
            
    if cv2.waitKey(1) & 0xFF == ord('f'):
        active = False if active else True
    if active:
        cv2.imwrite("data/train/null/null_"+str(count)+".jpg",x)
        print("cap count:",count)
        count+=1
        
    if cv2.waitKey(2) & 0xFF == ord(' '):
        break
cap.release()
cv2.destroyAllWindows()
  