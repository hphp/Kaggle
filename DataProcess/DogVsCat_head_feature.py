#!/usr/bin/python

'''
    written by hp_carrot
    2013-11-18
    load gray picture , resize to 25*25 , record in csv
    load gray picture , resize to 50*50, record in csv
'''
import cPickle 
import numpy
import pylab
from PIL import Image
import colorsys
import time
import os
import random
import transform_data_to_format as tdtf

DataHome = "../../data/Kaggle/DogVsCatData/"
train_feature_filename = DataHome + "DogVsCat_head_train_feature_2500.csv"
valid_feature_filename = DataHome + "DogVsCat_head_valid_feature_2500.csv"
print "going through the whole feature space to check if corresponding place is 1."
print "resizing"

def img_label(img_name):
    label = 0
    if (img_name[0] >= 'a') & (img_name[0] <= 'z'):
        label = 1
    elif (img_name[0] >= 'A') & (img_name[0] <= 'Z'):
        label = 0

    return label

img_name_list = os.listdir(DataHome + "head_images/")
features_list = []
for img_name in img_name_list:
    img = Image.open(open(DataHome + "head_images/" + img_name))
    img = img.resize((50,50),Image.ANTIALIAS)
    img = numpy.asarray(img, dtype='int32')
    feature= []
    feature.append(img_label(img_name))
    shape_len = len(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cur_v = 0
            if shape_len < 3:
                cur_v = img[i][j]
            else:
                for ch_v in img[i][j]:
                   cur_v += ch_v 
                cur_v /= 3
            feature.append(cur_v)

    features_list.append(feature)

print len(features_list)
train_index_list = random.sample(range(len(features_list)), len(features_list)/2 )
train_features_list = []
for i in train_index_list:
    train_features_list.append(features_list[i])
valid_features_list = []
for i in range(len(features_list)):
    if i in train_index_list:
        continue
    valid_features_list.append(features_list[i])

#print len(train_features_list)
#print len(valid_features_list)
tdtf.write_content_to_csv(train_features_list,train_feature_filename)
tdtf.write_content_to_csv(valid_features_list,valid_feature_filename)
