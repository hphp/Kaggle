#!/usr/bin/python

'''
    developed by hp_carrot
    2013-10-31
    a method that comes better than NearestNeighborsCentroid method 
    with about 90% precision , and with 90.18% test precision on Kaggle
'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import transform_data_to_format as tdtf

train_x , train_y = tdtf.read_data_to_ndarray("../data/train.csv",2000)
test_x = tdtf.read_test_data_to_ndarray("../data/test.csv",28000)
#valid_x , valid_y = tdtf.read_data_to_ndarray("../data/valid.csv",10000)
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(train_x,train_y)
#pred_train_y = clf.predict(train_x)
#pred_valid_y = clf.predict(valid_x)
pred_test_y = clf.predict(test_x)
'''
print("Classification report for classifier %s:\n%s\n"
      % (clf , metrics.classification_report(train_y , pred_train_y )))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(train_y , pred_train_y ))
print("Classification report for classifier %s:\n%s\n"
      % (clf , metrics.classification_report(valid_y , pred_valid_y )))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(valid_y , pred_valid_y ))
'''
tdtf.write_to_csv(pred_test_y,"../data/MNIST_KNearestNeighbors.out")
