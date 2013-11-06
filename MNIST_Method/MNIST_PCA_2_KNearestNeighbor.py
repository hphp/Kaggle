#!/usr/bin/python

'''
    developed by hp_carrot
    2013-11-03
    pca train data and test data to get more simple datas.
    precision of valid data : 92% , and only with 2000 train samples.
    test in kaggle with 91.17%... sucks.
'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import transform_data_to_format as tdtf
from sklearn import decomposition
import numpy

ori_train_x , train_y = tdtf.read_data_to_ndarray("../data/train.csv",42000)
ori_test_x = tdtf.read_test_data_to_ndarray("../data/test.csv",28000)
#ori_valid_x , valid_y = tdtf.read_data_to_ndarray("../data/valid.csv",100)

print "starting decomposition"
train_pca_component = decomposition.PCA(n_components=50).fit(ori_train_x)
transformed_train = train_pca_component.transform(ori_train_x)
reconstructed_train = train_pca_component.inverse_transform(transformed_train)
train_x = reconstructed_train

test_pca_component = decomposition.PCA(n_components=50).fit(ori_test_x)
transformed_test = test_pca_component.transform(ori_test_x)
reconstructed_test = test_pca_component.inverse_transform(transformed_test)
test_x = reconstructed_test
test_set_n = len(test_x)

'''
valid_pca_component = decomposition.PCA(n_components=100).fit(ori_valid_x)
transformed_valid = valid_pca_component.transform(ori_valid_x)
reconstructed_valid = valid_pca_component.inverse_transform(transformed_valid)
valid_x = reconstructed_valid
#valid_x = ori_train_x
'''

clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(train_x,train_y)
#pred_train_y = clf.predict(train_x)
#pred_valid_y = clf.predict(valid_x)
print "predicting"
predict_temp = []
pred_test_y = numpy.asarray(predict_temp)
for i in range(test_set_n/1000):
    pred_test_y = numpy.concatenate((pred_test_y,clf.predict(test_x[i*1000:min(test_set_n,(i+1)*1000)])))
    #print type(pp),pp.shape
    #predict_temp += pp
    print "each round 1000 , and this is the %d round" % (i+1)
'''
print("Classification report for classifier %s:\n%s\n"
      % (clf , metrics.classification_report(train_y , pred_train_y )))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(train_y , pred_train_y ))
print("Classification report for classifier %s:\n%s\n"
      % (clf , metrics.classification_report(valid_y , pred_valid_y )))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(valid_y , pred_valid_y ))
'''
tdtf.write_to_csv(pred_test_y,"../data/MNIST_PCA_2_KNearestNeighbors.out")
