# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 02:20:24 2017

@author: Arjun
"""
import os
#while(1):
#	i= input("press 1 to classify image\npress2 to retrain inception\n:")
#	if i=="1":
while(1):
	a = input("input image location for classification: ")
	os.system("python label_image.py --image="+a+" --graph=retrained_graph.pb --labels=retrained_labels.txt")
	input("press any key to continue")
#	elif i=="2":
#		os.system("python retrain.py --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --image_dir=D:\py\potato_or_not_potato\image_data")
		