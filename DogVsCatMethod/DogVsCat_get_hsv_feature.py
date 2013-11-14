#!/usr/bin/python

'''
    written by hp_carrot
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
import cPickle 
import numpy
import pylab
from PIL import Image
import colorsys
DataHome = "/home/hphp/Documents/data/Kaggle/DogVsCatData/"

def get_feature_index(l_id,l_N):
    index = 0
    for i in range(len(l_N)):
        index *= l_N[i]
        index += l_id[i]
    return index

print "resizing"
img = Image.open(open(DataHome + "train/" + "cat.5123.jpg"))
img = img.resize((250,250),Image.ANTIALIAS)
img_w , img_h = img.size
print img.size
img = numpy.asarray(img, dtype='float32') / 256
print img.shape
print img[0][0]
rgb = img[0][0]
print type(rgb) , rgb.shape , rgb
print "getting hsv"
hsv = colorsys.rgb_to_hsv(rgb[0],rgb[1],rgb[2])


print type(hsv),hsv
# float64 , float32 , float32
print "getting feature"
N = 5
CH = 2
CS = 2 
CV = 1
each_h_part = img_h / N
each_w_part = img_w / N
each_H_part = 100 / CH
each_S_part = 100 / CS
each_V_part = 100 / CV

features = []
# using this method , it cost O( CH*CS*CV*img_h*img_w )
# we can also go through the whole img , and fulfill its features.
print "first methoding"
for ih_c in range(CH):
    img_h_s = ih_c * each_H_part
    img_h_e = min ( ((ih_c+1) * each_H_part) , 100)
    for is_c in range(CS):
        img_s_s = is_c * each_S_part
        img_s_e = min ( ((is_c+1) * each_S_part) , 100)
        for iv_c in range(CV):
            img_v_s = iv_c * each_V_part
            img_v_e = min ( ((iv_c+1) * each_V_part) , 100)
            for h_c in range(N):
                h_s = h_c * each_h_part
                h_e = min((h_c + 1)*each_h_part,img_h)
                for w_c in range(N):
                   w_s = w_c * each_h_part
                   w_e = min((w_c+1)*each_w_part,img_w)
                   flag = 0
                   for hh in range(h_s,h_e):
                    if flag == 1:
                        break
                    for ww in range(w_s,w_e):
                        rgb = img[hh][ww]
                        img_h_v , img_s_v , img_v_v = colorsys.rgb_to_hsv(rgb[0],rgb[1],rgb[2])
                        img_h_v = numpy.float32(img_h_v) * 100. - 1
                        img_s_v = img_s_v * 100. - 1
                        img_v_v = img_v_v * 100. - 1
                      
                        if (img_h_v >= img_h_s) & (img_h_v < img_h_e ) \
                            & (img_s_v >= img_s_s) & (img_s_v < img_s_e ) \
                            & (img_v_v >= img_v_s) & (img_v_v < img_v_e ) :
                            flag = 1
                            break
                   features.append(flag)
print len(features) , type(features)
print features

print 'second methoding'
feature_2 = [0] * N * N * CH * CS * CV
for h_i in range(img_h):
    for w_i in range(img_w):
        rgb = img[h_i][w_i]
        img_h_v , img_s_v , img_v_v = colorsys.rgb_to_hsv(rgb[0],rgb[1],rgb[2])
        img_h_v = numpy.float32(img_h_v) * 100.
        img_s_v = img_s_v * 100.
        img_v_v = img_v_v * 100.

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
        feature_2[feature_index] = 1


print len(feature_2)
print feature_2
print features == feature_2

feature_file = open('color_feature.in','w')
cPickle.dump(feature_2,feature_file)
