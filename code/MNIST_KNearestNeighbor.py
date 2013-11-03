#!/usr/bin/python

'''
    developed by hp_carrot
    2013-10-31
    a method that comes better than NearestNeighborsCentroid method 
    with about 90% precision , and using 2000 training data , we get with 90.18% test precision on Kaggle
    modified by hp_carrot
    2013-11-03
    with 40000 training we get 96.514% score , which is good !!!!
'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import transform_data_to_format as tdtf

train_x , train_y = tdtf.read_data_to_ndarray("../data/train.csv",40000)
test_x = tdtf.read_test_data_to_ndarray("../data/test.csv",28000)
#valid_x , valid_y = tdtf.read_data_to_ndarray("../data/valid.csv",10000)
clf = KNeighborsClassifier(n_neighbors=10)
print "fitting"
clf.fit(train_x,train_y)
#pred_train_y = clf.predict(train_x)
#pred_valid_y = clf.predict(valid_x)
print "predicting"
pred_test_y = clf.predict(test_x)
'''
print("Classification report for classifier %s:\n%s\n"
      % (clf , metrics.classification_report(train_y , pred_train_y )))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(train_y , pred_train_y ))
print("Classification report for classifier %s:\n%s\n"
      % (clf , metrics.classification_report(valid_y , pred_valid_y )))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(valid_y , pred_valid_y ))
'''
print "writing to file"
tdtf.write_to_csv(pred_test_y,"../data/MNIST_KNearestNeighbors.out")
