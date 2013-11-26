#!/usr/bin/python

'''
    written by hp_carrot
    2013-11-26
    try 100*100
    2013-11-21
    add function to get feature of train data. , resize to 50*50, record in csv
    written by hp_carrot
    this method is stable to generate random features split into two valid/train files from src_img_route. to crspd files.
    2013-11-20
    rm load random pixel value.
    2013-11-20
    load random pixel value as feature. did not work well
    2013-11-18
    load gray picture , resize to 25*25 , record in csv
    load gray picture , resize to 50*50, record in csv
'''
import os
import sys
import random
import cv2
import transform_data_to_format as tdtf

DataHome = "../../data/Kaggle/DogVsCatData/"
src_img_route = "train/"
train_feature_filename = DataHome + "DogVsCat_train_feature_1w.csv"
valid_feature_filename = DataHome + "DogVsCat_valid_feature_1w.csv"
'''
src_img_route = "train/"
train_feature_filename = DataHome + "DogVsCat_train_feature_2500.csv"
valid_feature_filename = DataHome + "DogVsCat_valid_feature_2500.csv"
src_img_route = "head_images/"
train_feature_filename = DataHome + "DogVsCat_head_train_feature_2500.csv"
valid_feature_filename = DataHome + "DogVsCat_head_valid_feature_2500.csv"
'''
def img_label(img_name):
    label = 0
    if src_img_route == "train/":
        part_list = img_name.split('.')
        if part_list[0] == "dog":
            label = 1
        elif part_list[0] == "cat":
            label = 0
    elif src_img_route == "head_images/":
        if (img_name[0] >= 'a') & (img_name[0] <= 'z'):
            label = 1
        elif (img_name[0] >= 'A') & (img_name[0] <= 'Z'):
            label = 0

    return label

if len(sys.argv) > 1:
    DataHome = sys.argv[1]
if len(sys.argv) > 2:
    src_img_route = sys.argv[2]

features_list = []
img_name_list = os.listdir(DataHome + src_img_route)
start_index = 0
end_index = len(img_name_list)
if len(sys.argv) > 4:
    train_feature_filename = sys.argv[3]
    valid_feature_filename = sys.argv[4]
if len(sys.argv) > 6:
    start_index = int(sys.argv[5])
    end_index = int(sys.argv[6])
end_index = min(end_index, len(img_name_list))

for index in range(start_index, end_index):
    img_name = img_name_list[index]
    img = cv2.imread(DataHome + src_img_route + img_name)
    if (len(img.shape) > 2) & (img.shape[2] == 3):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # to resize 
    w,h=(100, 100)
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
# write / cover content to file
tdtf.append_content_to_csv(train_features_list,train_feature_filename)
tdtf.append_content_to_csv(valid_features_list,valid_feature_filename)
