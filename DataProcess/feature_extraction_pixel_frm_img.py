#!/usr/bin/python

'''
    written by hp_carrot
    2013-12-03
    this method is stable to generate pixel_value features from raw images.
    format pic_id features.
'''
import os
import sys
import cv2
import transform_data_to_format as tdtf

DataHome = "../../data/Kaggle/CIFAR-10/"
src_img_route = "train/"
feature_filename = DataHome + "train_feature_pixel_v.csv"
resize_to_img_h, resize_to_img_w = (32, 32)

def img_label(img_name):
    img_part = img_name.split('.')
    label = None
    if len(img_part) == 2:
        label = img_part[0]
    elif len(img_part) == 3:
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
    feature_filename = DataHome + sys.argv[3]

if len(sys.argv) > 5:
    resize_to_img_h = int(sys.argv[4])
    resize_to_img_w = int(sys.argv[5])

if len(sys.argv) > 7:
    start_index = int(sys.argv[6])
    end_index = int(sys.argv[7])
end_index = min(end_index, len(img_name_list))

for index in range(start_index, end_index):
    img_name = img_name_list[index]
    img = cv2.imread(DataHome + src_img_route + img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # to resize 
    w,h = (resize_to_img_w, resize_to_img_h)
    #print img.shape, w, h, img_name
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
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
# write / cover content to file
'''
tdtf.append_content_to_csv(features_list, feature_filename)
#tdtf.wr_content_to_csv(valid_features_list,valid_feature_filename)
