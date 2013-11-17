#!/usr/bin/python

'''
    2013-11-17
    1.read trained classifier from joblib.cPickle.
    2.read test_set_x
    3.predit
    4.scores:
    reading 125000 test_set for 0.77m
    loading from svm.svc trained joblib for 0.00m
    predicting and writting total for 30m
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
test_data_set_route = DataHome + "test.csv"

print "reading test data"
start_sec = time.time()
test_set = tdtf.read_csv_data_to_int_list(test_data_set_route,None,0)
test_set_x , test_set_y = test_set
print len(test_set_x)
end_sec = time.time()
print 'practical reading data time : %.2fm ' % ((end_sec - start_sec) / 60.)

start_sec = time.time()
print "loading svm classifier from joblib"
classifier = joblib.load(DataHome + 'svm_svc_pkl/svm.svc.pkl' , mmap_mode = 'c')
end_sec = time.time()
print 'practical loading svm time : %.2fm ' % ((end_sec - start_sec) / 60.)


start_sec = time.time()
print "predicting"
pred_test_y = classifier.predict(test_set_x)
end_sec = time.time()
print 'practical predicting time : %.2fm ' % ((end_sec - start_sec) / 60.)


start_sec = time.time()
print "predicting"
tdtf.write_to_csv(pred_test_y,DataHome + "DogVsCat.svm.svc.csv")
end_sec = time.time()
print 'practical writting to csv time : %.2fm ' % ((end_sec - start_sec) / 60.)
