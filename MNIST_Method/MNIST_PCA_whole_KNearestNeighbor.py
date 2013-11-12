#!/usr/bin/python

'''
developed by hp_carrot
2013-11-02
pca train data and valid data , and use the transformed features to calculate .
precision of valid data : 10-20 % , with 2000 train examples. , pca 100 - 400
component.
it is very terrible... seems after pca , physical distance get confused. 
'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import transform_data_to_format as tdtf
from sklearn import decomposition
import numpy

ori_train_x , train_y = tdtf.read_data_to_ndarray("../data/train.csv",2000)
ori_valid_x , valid_y = tdtf.read_data_to_ndarray("../data/valid.csv",100)
#ori_test_x = tdtf.read_test_data_to_ndarray("../data/test.csv",28)

print "starting decomposition"
pca_component_train = decomposition.PCA(n_components=100).fit(ori_train_x)
transformed_train_x = pca_component_train.transform(ori_train_x)
print transformed_train_x[0]
#pca_component_test = decomposition.PCA(n_components=100).fit(ori_test_x)
#transformed_test_x = pca_component_test.transform(ori_test_x)
pca_component_valid = decomposition.PCA(n_components=100).fit(ori_valid_x)
transformed_valid_x = pca_component_valid.transform(ori_valid_x)

print "starting KNeighborsClassify fitting"
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(transformed_train_x,train_y)

print "predicting"
pred_valid_y = clf.predict(transformed_valid_x)
print("Classification report for classifier %s:\n%s\n"
      % (clf , metrics.classification_report(valid_y , pred_valid_y )))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(valid_y , pred_valid_y ))
'''
pred_train_y = clf.predict(train_x)
#pred_test_y = clf.predict(test_x)
print("Classification report for classifier %s:\n%s\n"
      % (clf , metrics.classification_report(train_y , pred_train_y )))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(train_y , pred_train_y ))
#tdtf.write_to_csv(pred_test_y,"../data/MNIST_KNearestNeighbors.out")
'''
