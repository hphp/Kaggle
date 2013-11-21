#!/usr/bin/python

'''
    written by hp_carrot
    2013-11-20
    generate bg features
'''
import cPickle 
import numpy
import time
import os
import random
import cv2
import transform_data_to_format as tdtf

DataHome = "../../data/Kaggle/DogVsCatData/"
feature_filename = DataHome + "DogVsCat_bg_feature_2500.csv"
features_list = []
bg_name_list = os.listdir(DataHome + "bg/")
for bg_name in bg_name_list:
    bg = cv2.imread(DataHome + "bg/" + bg_name)
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    h, w = bg.shape
    #show the gray img
    #cv2.imshow("w2",img)
    #cv2.waitKey(0)

    #reshape (h,w) to (h*w,) 
    bg=bg.reshape(w*h) 
    feature= [2]
    for f_v in bg:
        feature.append(f_v)
    features_list.append(feature)

print len(features_list),len(features_list[0]),len(features_list[-1])
tdtf.write_content_to_csv(features_list,feature_filename)
'''
train_index_list = random.sample(range(len(features_list)), len(features_list)/2 )
train_features_list = []
for i in train_index_list:
    train_features_list.append(features_list[i])
valid_features_list = []
for i in range(len(features_list)):
    if i in train_index_list:
        continue
    valid_features_list.append(features_list[i])

print len(train_features_list)
print len(valid_features_list)
tdtf.write_content_to_csv(train_features_list,train_feature_filename)
'''
