'''
2013-11-18
by hp_carrot
run test from pickle_stored LogisticRegression OBJ
'''
import cPickle
import gzip
import os
import sys
import time

import numpy
import matplotlib.pyplot as plt
import csv


import theano
theano.config.floatX='float32'
import theano.tensor as T

if not "../DL_Method/" in sys.path:
    sys.path.append("../DL_Method/")
from logistic_sgd import LogisticRegression
import load_data
DataHome = "../../data/Kaggle/MNISTData/"
train_model_route = DataHome + "logistic_sgd_trained_model.pkl"

def sgd_predict(dataset=DataHome, \
                           batch_size=28):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """

    logistic_regression_model_pkl = open( train_model_route, 'r')
    logistic_regression_model_state = cPickle.load( logistic_regression_model_pkl )
    W, b = logistic_regression_model_state

    datasets = load_data.load_data(dataset)

    test_set_x, test_set_y = datasets[2]

    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    #print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                           # [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10, W=W, b=b)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch

    test_results = theano.function(inputs=[index],
            outputs= classifier.y_pred,
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size]})

    test_res = [test_results(i)
        for i in xrange(n_test_batches)]
    print test_res

if __name__ == '__main__':
    sgd_predict()
