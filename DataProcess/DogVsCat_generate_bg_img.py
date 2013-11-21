#!/usr/bin/python

'''
    written by hp_carrot
    2013-11-20
    generate background image.
'''
import cPickle 
import numpy
import os
import random
import cv2

DataHome = "../../data/Kaggle/DogVsCatData/"
head_name_list = os.listdir(DataHome + "head_images/")
head_img_n = len(head_name_list)
#head_img_n = 3
img_name_list = os.listdir(DataHome + "train/")
img_num = len(img_name_list)
img_name_index_list = random.sample(range(img_num), 2*head_img_n)
for index in img_name_index_list:
    img_name = img_name_list[index]
    img = cv2.imread(DataHome + "train/" + img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_h, img_w = img.shape
    if (img_h > 100) & (img_w > 100):
        img_h_s = random.randrange(50, img_h-50)
        img = img[img_h_s-50: img_h_s+50]
        img_w_s = random.randrange(50, img_w-50)
        img = numpy.hsplit(img, numpy.array([img_w_s-50, img_w_s+50]))[1]
    #print img.shape

    # to resize 
    w,h=(50,50)
    img = cv2.resize(img,(w,h),interpolation=cv2.INTER_LINEAR)

    #print "after resize and gray:",type(img),img.shape,img.dtype

    #show the gray img
    #cv2.imshow("w2",img)
    #cv2.waitKey(0)
    
    cv2.imwrite(DataHome+"bg/"+img_name, img)
