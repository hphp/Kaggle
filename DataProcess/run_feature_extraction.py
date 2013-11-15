#!/usr/bin/python

import os

def convolution_feature_extraction():
    piclist = os.listdir("/home/hphp/Documents/data/Kaggle/DogVsCatData/test1/")
    t_range = len(piclist)
    period = 1000
    total = int(t_range/period)
    print total
    for rr in range(200,total):
        start = rr * 1000
        end = min((rr+1)*1000,t_range)
        cmd = "python feature_extraction.py " + str(start) +  " " + str(end)
        os.system(cmd)



def color_HSV_feature_extraction():
    piclist = os.listdir("/home/hphp/Documents/data/Kaggle/DogVsCatData/train/")
    t_range = len(piclist)
    period = 1000
    total = int(t_range/period)
    print total
    for rr in range(total):
        start = rr * 1000
        end = min((rr+1)*1000,t_range)
        cmd = "python DogVsCat_get_hsv_feature.py " + str(start) +  " " + str(end)
        os.system(cmd)

color_HSV_feature_extraction() 
