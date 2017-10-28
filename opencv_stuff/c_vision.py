# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 14:40:31 2017

@author: Potato
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
#static image reading

img = cv2.imread('g2.jpg',cv2.IMREAD_GRAYSCALE)

#using waitkey so image doesnt come and disappear, leaving the window dead
cv2.imshow('window name',img)
#waitkey specifies time to show it in ms, 0 = infinite until keypress
cv2.waitKey(0)
cv2.destroyAllWindows()

#plt.imshow(img,cmap='gray',interpolation='bicubic')
#using gray coz matplotlib uses rbg while cv2 uses bgr or some shit
#plt.imshow(img,cmap='gray',interpolation='bilinear')
#you can plot stuff
#plt.plot([50,100],[80,100],'c',linewidth=5)
'''


'''
#webcam feed

#0 = first webcam
cap = cv2.VideoCapture(0)
#for outputting to a file, set codec, apparently not working cri
#codec = cv2.VideoWriter_fourcc(*'XVID')
#name,codec,framerate,size
#outfile = cv2.VideoWriter('output.avi',codec, 15.0,(640,480))

while True:
    #bool,feed
    ret, frame = cap.read()
    #convert color
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#    outfile.write(gray)
    #open 2 frames
    cv2.imshow('window_name',frame)
    cv2.imshow('gray_frame',gray)
    #for wrting to file

    #stop when q pressed, waitkey specifies time to show it in ms
    #(eg 1000 will show 1 frame per sec(each iteration=1sec)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#outfile.release()
cv2.destroyAllWindows()
'''

'''
#drawing
img =  cv2.imread('g2.jpg', cv2.IMREAD_COLOR)

#LINE start, end, color in bgr, line width
#cv2.line(img,(0,0),(283,176),(0,255,0),1)

#RECT, TL,BR,color,width
#cv2.rectangle(img, (65,15),(200,130),(0,255,0),2)

#CIRC center,radius,color, linewidth(-1 for fill)
#cv2.circle(img,(130,70),60,(0,0,255),2)

#polygon
#pts = np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
#reshape points to a 1x2 matrix
#pts = pts.reshape((-1,1,2))
#image, points, bool for closed or not,color,width
#cv2.polylines(img,[pts],True,(0,0,255),1)

#write
font = cv2.FONT_HERSHEY_SIMPLEX
#image,text,start,font,size,color,width,anti aliasing
cv2.putText(img,'potato',(0,150),font,0.7,(255,255,255),1,cv2.LINE_AA)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''




'''
#image operations
img = cv2.imread('g2.jpg',cv2.IMREAD_COLOR)
#img = arr[width][height][3](in case of rgb)

#region of image
#roi = img[100:150,100:150]

#delete section
#x array, y array
img[100:150,100:150]=[0,0,0]
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
#thresholding
img = cv2.imread('h.jpg')
img = cv2.resize(img,(640,360))
#ret, threshold = cv2.threshold(img,12,255,cv2.THRESH_BINARY)
grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret2, threshold2 = cv2.threshold(grayscaled,150,255,cv2.THRESH_BINARY)
#adaptive guassian threshold, img,maxval,
gaus = cv2.adaptiveThreshold(grayscaled,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)


cv2.imshow('original',img)
#cv2.imshow('threshold',threshold)
cv2.imshow('threshold2',threshold2)
cv2.imshow('gaus',gaus)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


'''
#color filtering
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #adjust lower hue in hsv for optimal threshold
    lower_hsv = np.array([150,0,100])
    upper_hsv = np.array([255,255,255])

    mask = cv2.inRange(hsv,lower_hsv,upper_hsv)
    #if it's in the mask range itll be one else 0
    res = cv2.bitwise_and(frame,frame,mask=mask)

    #smoothing/averaging
    #divide by 15x15=225,(if (5,5) then 25)
    #kernel = np.ones((15,15),np.float32)/225
    #smoothed = cv2.filter2D(mask,-1,kernel)

    #blurring
    #blur = cv2.GaussianBlur(smoothed,(15,15),0)

    #erosion and dilation(expanding,squeezing pixel distributions)
    #kernel = np.ones((5,5),np.uint8)
    #erosion = cv2.erode(mask, kernel, iterations=1)
    #dilation = cv2.dilate(mask, kernel, iterations=1)

    #opening/ remove false positives, closing/remove false negatives
    #opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN,kernel)
    #closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,kernel)

    #gradients/edge detection (cv_64f is the default datatype)
    #laplacian = cv2.Laplacian(frame,cv2.CV_64F)
    #sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
    #sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)

    #edge detection: img, thresholds
    edges = cv2.Canny(frame,100,200)

  #  cv2.imshow('frame',frame)
  #  cv2.imshow('mask',mask)
  #  cv2.imshow('res',res)
  #  cv2.imshow('smooth',smoothed)
  #  cv2.imshow('blur',blur)
  #  cv2.imshow('erosion',erosion)
  #  cv2.imshow('dilation',dilation)
  #  cv2.imshow('opening',opening)
  #  cv2.imshow('closing',closing)
  #  cv2.imshow('laplacian',laplacian)
  #  cv2.imshow('sobelx',sobelx)
  #  cv2.imshow('sobely',sobely)
    cv2.imshow('edges',edges)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
'''


'''
#template matching
img_bg = cv2.imread('h.jpg')
img_gray = cv2.cvtColor(img_bg,cv2.COLOR_BGR2GRAY)
template = cv2.imread('dial.jpg',0)
#get width,height, we're inverting array coz shape returns rows(height),cols(width)
w,h = template.shape[::-1]

#res is storing all the possible probabilities(yep)
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_bg,pt,(pt[0]+w, pt[1]+h),(0,255,0),1)
cv2.imshow('detected',img_bg)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''




#face recog using haar cascade classifier(haar cascades are features se..meh just go google)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #detect faces based on haar and return box coords
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    #plot all detected faces
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        #find smaller things like eyes within rois
        #roi_gray = gray[y:y+h,x:x+w]
        #roi_color = img[y:y+h,x:x+w]
        #search for eyes inside roi
        #eyes = eye_cascade.detectMultiScale(roi_gray)
    cv2.imshow('frames',img)
    k = cv2.waitKey(30) & 0xff
    #27 for escape key
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
