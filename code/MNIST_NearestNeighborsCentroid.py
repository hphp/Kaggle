#!/usr/bin/python
'''

developed by hp_carrot

2013-10-31

 specific analysis or data in MNIST_NearestNeighborsCentroid.anls

 this code use NearestNeighborsCentroid method , results shows no specific
 advancement. with about 89% precision or so.
'''
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import metrics
import numpy
import transform_data_to_format as tdtf

#train_x , train_y = tdtf.read_data_to_ndarray("../data/train.csv",42000)
#train_x , train_y = tdtf.read_data_to_ndarray("../data/train.csv",2100)
#valid_x , valid_y = tdtf.read_data_to_ndarray("../data/valid.csv",21000)
#test_x = tdtf.read_test_data_to_ndarray("../data/test.csv",28000);

clf = NearestCentroid()
clf.fit(train_x,train_y)

#NearestCentroid(metric='euclidean', shrink_threshold=None)
#pred_y = clf.predict(test_x)
#pred_train_y = clf.predict(train_x[0:21000])
pred_valid_y = clf.predict(valid_x)

#print pred_y

#tdtf.write_to_csv(pred_y,"../data/MNIST_NearestNeighborsCentroid.out")

#print("Classification report for classifier %s:\n%s\n"
#      % (clf , metrics.classification_report(train_y , pred_train_y )))
'''
print("Classification report for classifier %s:\n%s\n"
      % (clf , metrics.classification_report(train_y[0:21000] , pred_train_y )))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(train_y[0:21000] , pred_train_y ))
'''
print("Classification report for classifier %s:\n%s\n"
      % (clf , metrics.classification_report(valid_y , pred_valid_y )))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(valid_y , pred_valid_y ))

