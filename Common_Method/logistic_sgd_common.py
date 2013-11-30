#!/usr/bin/python

"""
modified by hp_carrot
2013-11-30
create to give a common method.
"""


__docformat__ = 'restructedtext en'

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

import sys;
if not "../DataProcess/" in sys.path:
    sys.path.append("../DataProcess/")
import transform_data_to_format as tdtf
if not "../DL_Method/" in sys.path:
    sys.path.append("../DL_Method/")
from logistic_sgd import LogisticRegression
import load_data

DataHome = "/home/hphp/Documents/data/Kaggle/DogVsCatData/"
train_dataset_route = DataHome + "DogVsCat_train_feature_1w.csv"
valid_dataset_route = DataHome + "DogVsCat_valid_feature_1w.csv"
train_model_route = DataHome + "DogVsCat_runtrained_model_ls.np.pkl"
layer0_input_shape = (100, 100)
layer0_output_shape = 2

def sgd_optimization_mnist(tr_start_index=1, tr_limit=3000, vl_start_index=1, vl_limit=3000,
                           learning_rate=0.0015, n_epochs=25,#000,
                           batch_size=3000):
    train_set = tdtf.read_data_patch_to_ndarray(train_dataset_route, tr_start_index, tr_limit)
    datasets = load_data.shared_dataset(train_set)
    train_set_x, train_set_y = datasets

    valid_set = tdtf.read_data_patch_to_ndarray(valid_dataset_route, vl_start_index, vl_limit)
    print valid_set[1]
    datasets = load_data.shared_dataset(valid_set)
    valid_set_x, valid_set_y = datasets

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    #n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

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
    in_shape = layer0_input_shape[0] * layer0_input_shape[1]
    rng = numpy.random.RandomState(23555)
    W_bound=1
    tmp_W = theano.shared(numpy.asarray(
            rng.uniform(low=0, high=W_bound, size=(in_shape, layer0_output_shape)), dtype=theano.config.floatX),
            borrow=True)
    classifier = LogisticRegression(input=x, n_in=in_shape, n_out=layer0_output_shape)
                                    #,W=tmp_W)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    '''
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    test_results = theano.function(inputs=[index],
            outputs= classifier.y_pred,
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})
    '''
    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index],
            outputs=[cost, classifier.errors(y)],
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    #print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()
    best_train_loss = numpy.inf

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost, train_err = train_model(minibatch_index)
            print('epoch %i, minibatch %i/%i, train_cost %f, train_error %.2f %%' % \
                (epoch, minibatch_index + 1, n_train_batches,
                minibatch_avg_cost,
                train_err* 100.))

            if best_train_loss > train_err:
                best_train_loss = train_err

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                    (epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set
                    layer_state = classifier.__getstate__()
                    trained_model_list = [layer_state]
                    trained_model_array = numpy.asarray(trained_model_list)
                    classifier_file = open(train_model_route, 'w')
                    cPickle.dump([1,2,3], classifier_file, protocol=2)
                    numpy.save(classifier_file, trained_model_array)
                    classifier_file.close()
                    '''
                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    test_res = [test_results(i)
                                   for i in xrange(n_test_batches)]

                    print(('     epoch %i, minibatch %i/%i, test error of best'
                       ' model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                         test_score * 100.))
                     '''

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%'
           'with best train_performance %f %%') %
                 (best_validation_loss * 100., test_score * 100., best_train_loss * 100.))
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))
    '''
    test_res = [test_results(i)
        for i in xrange(n_test_batches)]
    #print type(test_res)
    #print len(test_res)
    a = []
    for i in range(len(test_res)):
        for ele in test_res[i]:
            a.append(ele)
    #print len(a)

    #print a
    write_to_csv(a,'../data/pringle_2.csv')
    '''

if __name__ == '__main__':
    sgd_optimization_mnist()