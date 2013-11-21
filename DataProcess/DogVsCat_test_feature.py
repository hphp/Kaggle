#!/usr/bin/python

'''
    written by hp_carrot
    2013-11-21
    get test feature
'''
import os
import random
import cv2
import transform_data_to_format as tdtf

DataHome = "../../data/Kaggle/DogVsCatData/"
src_img_route = "test1/"
test_feature_filename = DataHome + "DogVsCat_test_feature_2500.csv"

def img_label(img_name):
    return int(img_name.split(".")[0])

features_list = []
img_name_list = os.listdir(DataHome + src_img_route)
for img_name in img_name_list:
    img = cv2.imread(DataHome + src_img_route + img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # to resize 
    w,h=(50,50)
    img = cv2.resize(img,(w,h),interpolation=cv2.INTER_LINEAR)
    #print "after resize and gray:",type(img),img.shape,img.dtype

    #show the gray img
    #cv2.imshow("w2",img)
    #cv2.waitKey(0)

    #reshape (h,w) to (h*w,) 
    img=img.reshape(w*h) 
    feature= []
    feature.append(img_label(img_name))
    for f_v in img:
        feature.append(f_v)
    features_list.append(feature)

print len(features_list),len(features_list[0]),len(features_list[-1])
tdtf.wr_content_to_csv(features_list,test_feature_filename)
