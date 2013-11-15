#!/usr/bin/python

'''
    written by hp_carrot
    2013-11-15

    2013-11-14
    load picture , resize , cut into N*N patches , get hsv values , load into features.

    Part 1:
    1.for each picture , resize . to 250*250 if bigger ,
        to 100*100 if smaller than 250 / or ignore smaller ones.
    2.cut to partions.
    3.check each partion if in [S_s,S_e]*[C_s,C_e]*[H_s,H_e]
    4.record in pickle/file
    2013-11-14
    19:34 fulfill without decoration
'''
import time
import os
import sys
import cPickle 
import numpy
import pylab
from PIL import Image
import colorsys
import transform_data_to_format as tdtf
DataHome = "/home/hphp/Documents/data/Kaggle/DogVsCatData/"

def get_feature_index(l_id,l_N):
    index = 0
    for i in range(len(l_N)):
        index *= l_N[i]
        index += l_id[i]
    return index

'''
def get_feature_numpy(img):
    img_w , img_h = img.size
    img = numpy.asarray(img, dtype='float32') / 256
    print "getting feature"
    N = 5
    CH = 10
    CS = 5
    CV = 5
    each_h_part = img_h / N
    each_w_part = img_w / N
    each_H_part = 100 / CH
    each_S_part = 100 / CS
    each_V_part = 100 / CV

    print 'getting features , second methoding'
    features = [0] * N * N * CH * CS * CV
    for h_i in range(img_h):
        for w_i in range(img_w):
            rgb = img[h_i][w_i]
            img_h_v , img_s_v , img_v_v = colorsys.rgb_to_hsv(rgb[0],rgb[1],rgb[2])
            img_h_v = (numpy.float32(img_h_v) * 100.)
            img_s_v = (img_s_v * 100.)
            img_v_v = (img_v_v * 100.)
            #print type(hsv),hsv,type(hsv[0])
            # float64 , float32 , float32

            h_part_i = int(h_i / each_h_part)
            w_part_i = int(w_i / each_w_part)
            H_part_i = int((img_h_v-1) / each_H_part)
            S_part_i = int((img_s_v-1) / each_S_part)
            V_part_i = int((img_v_v-1) / each_V_part)

            list_id = ( \
                      H_part_i \
                    , S_part_i \
                    , V_part_i \
                    , h_part_i \
                    , w_part_i \
                    )
            list_N = (CH,CS,CV,N,N)
            feature_index = get_feature_index(list_id,list_N)
            if feature_index >= len(features):
                print "h,w= " , h_i,w_i, ", rgb = " , rgb , ",hsv = " , img_h_v,img_s_v,img_v_v , ",index = " , feature_index
            else:
                features[feature_index] = 1
    return features
'''

def get_feature(img):
    img_w , img_h = img.size
    #img = numpy.asarray(img, dtype='float32') / 256
    img = list(img.getdata())
    #print img.shape
    #print img[0][0]
    #rgb = float(img[0][0])
    #print "rgb type " , type(rgb) , rgb.shape , rgb
    #print "getting hsv"
    #hsv = colorsys.rgb_to_hsv(rgb[0],rgb[1],rgb[2])


    #print type(hsv),hsv,type(hsv[0])
    # float64 , float32 , float32
    #print "getting feature"
    N = 5
    CH = 10
    CS = 5
    CV = 5
    each_h_part = img_h / N
    each_w_part = img_w / N
    each_H_part = 100 / CH
    each_S_part = 100 / CS
    each_V_part = 100 / CV

    #print 'getting features , second methoding'
    features = [0] * N * N * CH * CS * CV
    for h_i in range(img_h):
        for w_i in range(img_w):
            rgb_o = img[h_i * img_w + w_i]
            rgb = []
            for ele in rgb_o:
                ele /= 256.
                rgb.append(ele)
            #rgb = img[h_i][w_i]
            img_h_v , img_s_v , img_v_v = colorsys.rgb_to_hsv(rgb[0],rgb[1],rgb[2])
            img_h_v = (img_h_v * 100.)
            #img_h_v = (numpy.float32(img_h_v) * 100.)
            img_s_v = (img_s_v * 100.)
            img_v_v = (img_v_v * 100.)
            #print "type of img_s_v " , type(img_s_v)

            h_part_i = int(h_i / each_h_part)
            w_part_i = int(w_i / each_w_part)
            H_part_i = int((img_h_v-1) / each_H_part)
            S_part_i = int((img_s_v-1) / each_S_part)
            V_part_i = int((img_v_v-1) / each_V_part)

            list_id = ( \
                      H_part_i \
                    , S_part_i \
                    , V_part_i \
                    , h_part_i \
                    , w_part_i \
                    )
            list_N = (CH,CS,CV,N,N)
            feature_index = get_feature_index(list_id,list_N)
            if feature_index >= len(features):
                print "h,w= " , h_i,w_i, ", rgb = " , rgb , ",hsv = " , img_h_v,img_s_v,img_v_v , ",index = " , feature_index
            else:
                features[feature_index] = 1
    return features

piclist = os.listdir(DataHome + "train/")
train_list = []
start = 0
end = len(piclist)
if len(sys.argv) > 2:
    start = int(sys.argv[1])
    end = int(sys.argv[2])
train_set = []
for i in range(start,end): #len(piclist)):
    img_route = piclist[i]
    img_route_list = img_route.split(".")
    breed = 1 
    if img_route_list[0] == "cat":
        breed = 0
    else:
        breed = 1
    img = Image.open(open(DataHome + "train/" + img_route))
    img_w , img_h = img.size
    if (img_w < 250) | (img_h < 250) :
        continue
    #print "resizing"
    img = img.resize((250,250),Image.ANTIALIAS)
    features = get_feature(img)
    breed_list = [breed]
    img_info = breed_list + features
    #print type(img_info),len(img_info)
    train_set.append(img_info)
print "writing from %d to %d " % (start,end)
#print train_set
tdtf.write_content_to_csv(train_set,DataHome + "train.csv" )
#print len(features)
#print features
#print features == features
#feature_file = open('color_feature.cPickle','w')
#cPickle.dump(features,feature_file)
