#!/usr/bin/python

'''
    developed by hp_carrot
    2013-11-01
    only pca traindata to get more simple datas.
    precision of valid data : 90% , and only with 2000 train samples.
'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import transform_data_to_format as tdtf
from sklearn import decomposition
import numpy

ori_train_x , train_y = tdtf.read_data_to_ndarray("../data/train.csv",2000)
temp_x = []

print "starting decomposition"
pca_component = decomposition.PCA(n_components=100).fit(ori_train_x)
for otx_ele in ori_train_x :
    transformed = pca_component.transform(otx_ele)
    reconstructed = pca_component.inverse_transform(transformed)
    temp_x.append(reconstructed[0])
train_x = numpy.asarray(temp_x)

#test_x = tdtf.read_test_data_to_ndarray("../data/test.csv",280)
valid_x , valid_y = tdtf.read_data_to_ndarray("../data/valid.csv",2000)
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(train_x,train_y)
pred_train_y = clf.predict(train_x)
pred_valid_y = clf.predict(valid_x)
#pred_test_y = clf.predict(test_x)
print("Classification report for classifier %s:\n%s\n"
      % (clf , metrics.classification_report(train_y , pred_train_y )))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(train_y , pred_train_y ))
print("Classification report for classifier %s:\n%s\n"
      % (clf , metrics.classification_report(valid_y , pred_valid_y )))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(valid_y , pred_valid_y ))
#tdtf.write_to_csv(pred_test_y,"../data/MNIST_KNearestNeighbors.out")
