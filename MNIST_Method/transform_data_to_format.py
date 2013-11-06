#!/usr/bin/env python

'''
    developed by xcbfreedom
    2013-10-28
    transform data between csv ，numpy.ndarray , files.
'''

import Image
import numpy as np
import matplotlib.pyplot as plt
import csv

def write_to_csv(pred_list,filename):
    csv_writer = csv.writer(open(filename, "wb"), delimiter=",")
    csv_writer.writerow(['ImageId','Label'])
    index=1
    for row in pred_list:
        csv_writer.writerow([index,row])
        index+=1

def read_test_data_to_ndarray(filname="../data/test.csv", limit=None):
    print "Reading data from %s " % filname
    data = []
    csv_reader = csv.reader(open(filname, "r"), delimiter=",")
    index = 0
    for row in csv_reader:
        index += 1
        if index == 1:
            continue
        data.append(np.float32(row)/255)
        if limit != None and index == limit + 1:
            break
    data_x=np.asarray(data)
    #print data_x,data_x.shape,data_x.dtype,type(data_x)
    return data_x

def read_data_to_ndarray(filname="../data/train.csv", limit=None):

    print "Reading data %s" % filname
    data = []
    labels = []
    csv_reader = csv.reader(open(filname, "r"), delimiter=",")
    index = 0
    for row in csv_reader:
        index += 1
        if index == 1:
            continue
        labels.append(int(row[0]))
        row = row[1:]
        data.append(np.float32(row)/255)
        if limit != None and index == limit + 1:
            break
    data_x=np.asarray(data)
    data_y=np.asarray(labels,dtype=np.int32)
    return (data_x, data_y)


