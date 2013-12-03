#!/usr/bin/python

'''
    written by hp_carrot
    2013-11-20
    select and combine features from diff feature.csv
'''
import numpy
import random
import transform_data_to_format as tdtf

DataHome = "../../data/Kaggle/DogVsCatData/"
to_train_feature_filename = DataHome + "DogVsCat_head_train_feature_3c_2500.csv"
to_valid_feature_filename = DataHome + "DogVsCat_head_valid_feature_3c_2500.csv"
from_train_feature_filename = DataHome + "DogVsCat_head_train_feature_2500.csv"
from_valid_feature_filename = DataHome + "DogVsCat_head_valid_feature_2500.csv"
bg_feature_filename = DataHome + "DogVsCat_bg_feature_2500.csv"

from_feature_fname_list = [bg_feature_filename, from_train_feature_filename, from_valid_feature_filename]
to_feature_fname_list = [to_train_feature_filename, to_valid_feature_filename]

DataHome = "../../data/Kaggle/CIFAR-10/"
to_train_feature_filename = DataHome + "CIFAR_train_feature.csv"
to_valid_feature_filename = DataHome + "CIFAR_valid_feature.csv"
from_feature_filename = DataHome + "train_feature_Cls_pixelv.csv"

from_feature_fname_list = [from_feature_filename]
to_feature_fname_list = [to_train_feature_filename, to_valid_feature_filename]

features_list = []
for fname in from_feature_fname_list:
    t_f_list = tdtf.read_feature_from_csv(filname=fname, limit=10, header_n=0)
    features_list += (t_f_list)

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
print len(valid_features_list),len(valid_features_list[0])
#tdtf.wr_content_to_csv(train_features_list,to_train_feature_filename)
#tdtf.wr_content_to_csv(valid_features_list,to_valid_feature_filename)
