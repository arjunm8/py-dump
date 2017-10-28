# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 18:48:09 2017

@author: Arjun
"""

from collections import Counter
import os

text = "potatoes are pretty cool"

def word_counter(text):
    word_counts={}
    text = text.lower()
    ch = [",",".","'",'"',"!","?","!"]
    for c in ch:
        text = text.replace(c,"")
    word_counts = Counter(text.split(" "))
    return word_counts


def word_counter_potato(text):
    word_counts={}
    text = text.lower()
    ch = [",",".","'",'"',"!","?","!"]
    for c in ch:
        text.replace(c,"")
    for word in text.split(" "):
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    return word_counts


def read_book(title_path):
    with open(title_path,"r","utf-8") as current_file:
        text = current_file.read()
        text = text.replace("\n","").replace("\n","")
    return text

def word_stats(word_counts):
    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique,counts)

