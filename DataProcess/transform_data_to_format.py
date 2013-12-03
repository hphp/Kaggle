#!/usr/bin/env python

'''
developed by xcbfreedom
2013-10-28
transform data between csv,numpy.ndarray,files.
'''

import Image
import numpy as np
import matplotlib.pyplot as plt
import csv

def wr_content_to_csv(content_list,filename):
    csv_writer = csv.writer(open(filename, "w"), delimiter=",")
    for row in content_list:
        csv_writer.writerow(row)

def append_content_to_csv(content_list,filename):
    csv_writer = csv.writer(open(filename, "a"), delimiter=",")
    for row in content_list:
        csv_writer.writerow(row)

def write_content_to_csv(content_list,filename):
    csv_writer = csv.writer(open(filename, "a"), delimiter=",")
    for row in content_list:
        csv_writer.writerow(row)

def wr_to_csv(header, id_list, pred_list, filename):
    csv_writer = csv.writer(open(filename, "wb"), delimiter=",")
    csv_writer.writerow(header)
    index=0
    for row in pred_list:
        csv_writer.writerow([id_list[index],row])
        index+=1

def write_to_csv(pred_list,filename):
    csv_writer = csv.writer(open(filename, "wb"), delimiter=",")
    csv_writer.writerow(['ImageId','Label'])
    index=1
    for row in pred_list:
        csv_writer.writerow([index,row])
        index+=1

def read_test_data_xy_to_ndarray(filname="../data/test.csv", limit=None):
    print "Reading data from %s " % filname
    data = []
    labels = []
    csv_reader = csv.reader(open(filname, "r"), delimiter=",")
    index = 0
    for row in csv_reader:
        index += 1
        if index == 1:
            continue
        data.append(np.float32(row)/255)
        labels.append(10)
        if limit != None and index == limit + 1:
            break
    data_x=np.asarray(data)
    data_y=np.asarray(labels,dtype=np.int32)
    #print data_x,data_x.shape,data_x.dtype,type(data_x)
    return ( data_x , data_y )

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

def read_data_patch_to_ndarray(filname="../data/train.csv", start_line=0, limit=None):

    print "Reading data %s of lines %s , starting from %d " % (filname, str(limit), start_line)
    data = []
    labels = []
    csv_reader = csv.reader(open(filname, "r"), delimiter=",")
    index = -1
    for row in csv_reader:
        index += 1
        if index < start_line:
            continue
        if limit != None:
            if index >= start_line + limit:
                break
        labels.append(int(row[0]))
        row = row[1:]
        data.append(np.float32(row)/255)

    data_x=np.asarray(data)
    data_y=np.asarray(labels,dtype=np.int32)
    return (data_x, data_y)

def read_data_to_ndarray(filname="../data/train.csv", limit=None, header_n=1):

    print "Reading data %s" % filname
    data = []
    labels = []
    csv_reader = csv.reader(open(filname, "r"), delimiter=",")
    index = 0
    for row in csv_reader:
        index += 1
        if index <= header_n :
            continue
        labels.append(int(row[0]))
        row = row[1:]
        data.append(np.float32(row)/255)
        if limit != None and index == limit + header_n:
            break
    data_x=np.asarray(data)
    data_y=np.asarray(labels,dtype=np.int32)
    return (data_x, data_y)

def read_csv_data_to_ndarray(filname="../data/train.csv", limit=None):

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
        data.append(np.float32(row))
        if limit != None and index == limit + 1:
            break
    data_x=np.asarray(data)
    data_y=np.asarray(labels,dtype=np.int32)
    return (data_x, data_y)

def read_csv_data_to_int_list(filname="train.csv", limit=None, header_n = 1):
    # header_n = line numbers of header.

    print "Reading data %s" % filname
    data = []
    labels = []
    csv_reader = csv.reader(open(filname, "r"), delimiter=",")
    index = 0
    for row in csv_reader:
        index += 1
        if (header_n == 1) & (index <= header_n ):
            continue
        labels.append(int(row[0]))
        row = map ( int , (row[1:]))
        data.append(row)
        if limit != None and index == limit + header_n:
            break
    data_x=data
    data_y=labels
    return (data_x, data_y)

def read_feature_from_csv(filname="train.csv", limit=None, header_n = 1):
    # header_n = line numbers of header.

    print "Reading data %s" % filname
    data = []
    csv_reader = csv.reader(open(filname, "r"), delimiter=",")
    index = 0
    for row in csv_reader:
        index += 1
        if (header_n == 1) & (index <= header_n ):
            continue
        row = map ( int , (row[0:]))
        data.append(row)
        if limit != None and index == limit + header_n:
            break
    return data
def read_s_feature_from_csv(filname="train.csv", limit=None, header_n = 1):
    # header_n = line numbers of header.

    print "Reading data %s" % filname
    data = []
    csv_reader = csv.reader(open(filname, "r"), delimiter=",")
    index = 0
    for row in csv_reader:
        index += 1
        if (header_n == 1) & (index <= header_n ):
            continue
        row = map ( str, (row[0:]))
        data.append(row)
        if limit != None and index == limit + header_n:
            break
    return data
