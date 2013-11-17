#!/usr/bin/python

'''
    2013-11-17
    using HSV-color-Channel-region as features, and 50*50 pixels each patch, train all 25000 images.
    the test accurancy comes to 50.9 % 
    it is not good.
    2013-11-15
    written by hp_carrot
    Part4:
    1.readin color_features # /texture_features
    2.SVM training
    3.record W,b in pickles.
'''
import cPickle
import gzip
import os
import sys
import time

import numpy
from sklearn import svm
from sklearn.externals import joblib


if not "../DataProcess/" in sys.path:
    sys.path.append("../DataProcess/")
import transform_data_to_format as tdtf

DataHome = "/home/hphp/Documents/data/Kaggle/DogVsCatData/"
train_data_set_route = DataHome + "train.csv"
test_data_set_route = DataHome + "test.csv"

train_set = tdtf.read_csv_data_to_int_list(train_data_set_route)
train_set_x , train_set_y = train_set
#print type(train_set_x),len(train_set_x),type(train_set_x[0]),len(train_set_x[0]),type(train_set_x[0][0])
#print type(train_set_y),len(train_set_y),type(train_set_y[0])
# <type 'list'> 20 <type 'list'> 6250 <type 'str'>
# <type 'list'> 20 <type 'int'>

classifier = svm.SVC()
classifier.fit(train_set_x,train_set_y)
#clf_file = open("svm.svc.cPickle","w")
#clf_pickle = cPickle.dump(classifier,clf_file)
clf_pickle = joblib.dump(classifier,DataHome + 'svm_svc_pkl/svm.svc.pkl')
