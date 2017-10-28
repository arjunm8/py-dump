# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 09:45:00 2017

@author: Arjun
"""

FILENAME = "potato_test"

with open(FILENAME, encoding = "utf-8") as f:
    a = f.readlines()

start_strip_ind = 0
end_strip_ind = 0
for i in range(len(a)):
    if "Comments Comments on" in a[i]:
        start_strip_ind = i
    #string occurs twice(start&end), capture end and strip
    if "Follow Follow"  in a[i]:
        end_strip_ind = i
        
del a[end_strip_ind:]
del a[:start_strip_ind]

count = 0
c = {}
temp = "potato"
#the 16 might need adjustments
for i in range(len(a)-16):
    if ("<https://" in a[i][:10]) and ("arjunm8" not in a[i]) and (a[i] != temp):
        #multiline desig loophole
        if(a[i+4]!="\n"):
            c[a[i-2].strip()] = {"id":a[i].strip(),"desig":a[i+3].strip().strip(a[i-2].strip())+" "+a[i+4].strip(),"contact":a[i+8]}
        else:
            c[a[i-2].strip()] = {"id":a[i].strip(),"desig":a[i+3].strip().strip(a[i-2].strip()),"contact":a[i+7]}
        
        print(a[i-2].strip())
        count+=1
        temp = a[i]
print("total names: ",count)
con_count = 0
rogue_comments = []
for i in c:
    if "9" in c[i]["contact"] or "8" in c[i]["contact"] or "7" in c[i]["contact"]:
        con_count+=1
    else:
        rogue_comments.append((i,c[i]))
print("with contacts: ",con_count)
print("rogue comments: ",len(rogue_comments))
