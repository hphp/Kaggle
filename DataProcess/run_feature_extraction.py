#!/usr/bin/python

'''
    written by hp_carrot
    2013-11-26
    add resized_pixel_fe()
'''

import os

def convolution_feature_extraction():
    piclist = os.listdir("/home/hphp/Documents/data/Kaggle/DogVsCatData/test1/")
    t_range = len(piclist)
    period = 1000
    total = int(t_range/period)
    print total
    for rr in range(200,total):
        start = rr * 1000
        end = min((rr+1)*1000,t_range)
        cmd = "python feature_extraction.py " + str(start) +  " " + str(end)
        os.system(cmd)



def color_HSV_feature_extraction(DataHome,img_data_dir,data_csv_file):
    piclist = os.listdir(DataHome + img_data_dir)
    t_range = len(piclist)
    period = 1000
    total = int(t_range/period) + 1
    print total
    for rr in range(total):
        start = rr * 1000
        end = min((rr+1)*1000,t_range)
        if start >= end :
            break
        cmd = "python DogVsCat_get_hsv_feature.py " + str(start) +  " " + str(end) + " " + img_data_dir + " " + data_csv_file
        print cmd
        os.system(cmd)

def resized_pixel_fe(DataHome, src_img_route, train_feature_filename, valid_feature_filename):

    piclist = os.listdir(DataHome + src_img_route)
    t_range = len(piclist)
    period = 1000
    total = int(t_range/period) + 1
    print total
    for rr in range(total):
        start = rr * 1000
        end = min((rr+1)*1000,t_range)
        if start >= end :
            break
        cmd = "python DogVsCat_patchtrain_feature.py " + DataHome + " " + src_img_route + " " + train_feature_filename + " " + valid_feature_filename + " " + str(start) +  " " + str(end)
        print cmd
        os.system(cmd)

def g_resized_pixel_fe(cmd_part1, t_range, period):

    total = int(t_range/period) + 1
    print total
    for rr in range(total):
        start = rr * period
        end = min((rr+1)*period, t_range)
        if start >= end :
            break
        cmd = cmd_part1 + " " + str(start) +  " " + str(end)
        print cmd
        os.system(cmd)

piclist = os.listdir("/home/hphp/Documents/data/Kaggle/CIFAR-10/train/")
t_range = len(piclist)
g_resized_pixel_fe("python feature_extraction_pixel_frm_img.py /home/hphp/Documents/data/Kaggle/CIFAR-10/ train/ train_feature_pixel_v.csv 32 32", t_range, 1000)
#DogVsCat_DataHome = "/home/hphp/Documents/data/Kaggle/DogVsCatData/"
#resized_pixel_fe(DogVsCat_DataHome, "train/", DogVsCat_DataHome+"DogVsCat_train_feature_1w.csv", DogVsCat_DataHome+"DogVsCat_valid_feature_1w.csv")
#color_HSV_feature_extraction(DogVsCat_DataHome,"test1/","test.csv")
#color_HSV_feature_extraction(DogVsCat_DataHome,"train/","train.csv")
