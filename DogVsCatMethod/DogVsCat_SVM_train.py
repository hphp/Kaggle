#!/usr/bin/python

'''
    2013-11-15
    written by hp_carrot
    Part4:
    1.readin color_features/texture_features
    2.SVM training
    3.record W,b in pickles.

    Part5:
    0.readin texture tiles info , [W,b] ,
    1.for each test picture , resize , and patch up.
    2.extract features for each part. color features / texture features.
    3.warm up SVM using existing W,b
    4.test if true.
'''
import cPickle
import gzip
import os
import sys
import time

import numpy
from sklearn import svm

if not "../DataProcess/" in sys.path:
    sys.path.append("../DataProcess/")
#import load_data
import transform_data_to_format as tdtf

DataHome = "/home/hphp/Documents/data/Kaggle/DogVsCatData/"
train_data_set_route = DataHome + "train.csv"

train_set = tdtf.read_csv_data_to_int_list(train_data_set_route)
train_set_x , train_set_y = train_set
#print type(train_set_x),len(train_set_x),type(train_set_x[0]),len(train_set_x[0]),type(train_set_x[0][0])
#print type(train_set_y),len(train_set_y),type(train_set_y[0])
# <type 'list'> 20 <type 'list'> 6250 <type 'str'>
# <type 'list'> 20 <type 'int'>

classifier = svm.SVC()
classifier.fit(train_set_x,train_set_y)
clf_file = open("svm.svc.cPickle","w")
clf_pickle = cPickle.dump(classifier,clf_file)


