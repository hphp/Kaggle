#!/usr/bin/python

'''
    written by hp_carrot
    2013-11-15
    1.check how many pics are smaller than 250*250
        2212
    2.test if small img could get resized to 250*250 imgs.
        tested ok , all 2212 imges resized without any trouble.
'''
import os
from PIL import Image
import sys

DataHome = "/home/hphp/Documents/data/Kaggle/DogVsCatData/test1/"

def count_small_img_number():
    piclist = os.listdir(DataHome)
    train_list = []
    start = 0
    end = len(piclist)
    print "input [start_index , end_index) , and data_home"
    if len(sys.argv) > 3:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
        DataHome = sys.argv[3]

    print DataHome
    train_set = []
    count = 0
    for i in range(start,end): #len(piclist)):
        img_route = piclist[i]
        img_route_list = img_route.split(".")
        img = Image.open(open(DataHome + img_route))
        img_w , img_h = img.size
        if (img_w < 250) | (img_h < 250) :
            print i,img_route,img.size
            count += 1

    print "total image with smaller size number : " , count

def test_if_small_img_could_resize_bigger():

    global DataHome
    print "hi", DataHome
    piclist = os.listdir(DataHome)
    train_list = []
    start = 0
    end = len(piclist)
    print "input [start_index , end_index) , and data_home"
    if len(sys.argv) > 3:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
        DataHome = sys.argv[3]

    print DataHome
    train_set = []
    count = 0
    for i in range(start,end): #len(piclist)):
        img_route = piclist[i]
        img_route_list = img_route.split(".")
        img = Image.open(open(DataHome + img_route))
        img_w , img_h = img.size
        if (img_w < 250) | (img_h < 250) :
            img = img.resize((250,250),Image.ANTIALIAS)
            count += 1
    print count

test_if_small_img_could_resize_bigger()
