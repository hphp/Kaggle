#!/usr/bin/python

'''
    2013-11-17
    written by hp_carrot
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
from sklearn.externals import joblib


if not "../DataProcess/" in sys.path:
    sys.path.append("../DataProcess/")
#import load_data
import transform_data_to_format as tdtf

DataHome = "/home/hphp/Documents/data/Kaggle/DogVsCatData/"
test_data_set_route = DataHome + "test.csv"

print "reading test data"
start_sec = time.time()
test_set = tdtf.read_csv_data_to_int_list(test_data_set_route,None,0)
test_set_x , test_set_y = test_set
print len(test_set_x)
end_sec = time.time()
print 'practical reading data time : %.2fm ' % ((end_sec - start_sec) / 60.)
#print type(train_set_x),len(train_set_x),type(train_set_x[0]),len(train_set_x[0]),type(train_set_x[0][0])
#print type(train_set_y),len(train_set_y),type(train_set_y[0])
# <type 'list'> 20 <type 'list'> 6250 <type 'str'>
# <type 'list'> 20 <type 'int'>
start_sec = time.time()
print "loading svm classifier from joblib"
classifier = joblib.load(DataHome + 'svm_svc_pkl/svm.svc.pkl' , mmap_mode = 'c')
end_sec = time.time()
print 'practical loading svm time : %.2fm ' % ((end_sec - start_sec) / 60.)


start_sec = time.time()
print "predicting"
pred_test_y = classifier.predict(test_set_x)
tdtf.write_to_csv(pred_test_y,DataHome + "DogVsCat.svm.svc.csv")
#print type(test_predict_y)
end_sec = time.time()
print 'practical predicting time : %.2fm ' % ((end_sec - start_sec) / 60.)
