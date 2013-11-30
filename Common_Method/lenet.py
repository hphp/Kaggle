#!/usr/bin/python
'''
written by hp_carrot
2013-11-27
try DogVsCat on convolution network
'''

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
theano.config.floatX='float32'
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

if not "../DL_Method/" in sys.path:
    sys.path.append("../DL_Method/")
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from convolutional_mlp import LeNetConvPoolLayer
import load_data
if not "../DataProcess/" in sys.path:
    sys.path.append("../DataProcess/")
import transform_data_to_format as tdtf

DataHome = "../../data/Kaggle/DogVsCatData/"
train_dataset_route = DataHome + "DogVsCat_train_feature_1w.csv"
valid_dataset_route = DataHome + "DogVsCat_valid_feature_1w.csv"
train_model_route = DataHome + "DogVsCat_runtrained_model_lenet.np.pkl"
if_trained_yet = False

rng = numpy.random.RandomState(23455)
nkerns=[20, 50]
batch_size = 100 

layer0_input_img_size = (100, 100) # ishape
filter0_shape = (20, 20)
layer1_input_img_size = (int((layer0_input_img_size[0]+1-filter0_shape[0])/2), int((layer0_input_img_size[1]+1-filter0_shape[1])/2))
filter1_shape = (15, 15)
layer2_input_img_size = (int((layer1_input_img_size[0]+1-filter1_shape[0])/2), int((layer1_input_img_size[1]+1-filter1_shape[1])/2))

N_OUT = 2 

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')   # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
                    # [int] labels

# Reshape matrix of rasterized images of shape (1, 50*50)
# to a 4D tensor, compatible with our LeNetConvPoolLayer
layer0_input = x.reshape((batch_size, 1, layer0_input_img_size[0], layer0_input_img_size[1]))

# Construct the first convolutional pooling layer:
# filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
# maxpooling reduces this further to (24/2,24/2) = (12,12)
# 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
layer0 = LeNetConvPoolLayer(rng, input=layer0_input, \
        image_shape=(batch_size, 1, layer0_input_img_size[0], layer0_input_img_size[1]), \
        filter_shape=(nkerns[0], 1, filter0_shape[0], filter0_shape[1]), poolsize=(2, 2) \
        )

# Construct the second convolutional pooling layer
# filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
# maxpooling reduces this further to (8/2,8/2) = (4,4)
# 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
        image_shape=(batch_size, nkerns[0], layer1_input_img_size[0], layer1_input_img_size[1]),
        filter_shape=(nkerns[1], nkerns[0], filter1_shape[0], filter1_shape[1]), poolsize=(2, 2) \
        )

# the TanhLayer being fully-connected, it operates on 2D matrices of
# shape (batch_size,num_pixels) (i.e matrix of rasterized images).
# This will generate a matrix of shape (20,32*4*4) = (20,512)
layer2_input = layer1.output.flatten(2)

# construct a fully-connected sigmoidal layer
layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * layer2_input_img_size[0] * layer2_input_img_size[1],
                     n_out=100, activation=T.tanh \
                     )

# classify the values of the fully-connected sigmoidal layer
layer3 = LogisticRegression(input=layer2.output, n_in=100, n_out=N_OUT\
                            )

def load_trained_model():
    global train_model_route 
    global layer0_input 
    global layer0 
    global layer1 
    global layer2_input 
    global layer2 
    global layer3 

    global layer0_input_img_size # ishape
    global filter0_shape
    global layer1_input_img_size
    global filter1_shape
    global layer2_input_img_size

    print "loading trained model"
    trained_model_pkl = open(train_model_route, 'r')
    trained_model_state_list = cPickle.load(trained_model_pkl)
    trained_model_state_array = numpy.load(trained_model_pkl)
    layer0_state, layer1_state, layer2_state, layer3_state = trained_model_state_array

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... loading the model'

    # Reshape matrix of rasterized images of shape (1, 50*50)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, layer0_input_img_size[0], layer0_input_img_size[1]))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input, \
            image_shape=(batch_size, 1, layer0_input_img_size[0], layer0_input_img_size[1]), \
            filter_shape=(nkerns[0], 1, filter0_shape[0], filter0_shape[1]), poolsize=(2, 2), \
                W=layer0_state[0], b=layer0_state[1] \
            )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], layer1_input_img_size[0], layer1_input_img_size[1]),
            filter_shape=(nkerns[1], nkerns[0], filter1_shape[0], filter1_shape[1]), poolsize=(2, 2), \
            W=layer1_state[0], b=layer1_state[1] \
            )

    # the TanhLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * layer2_input_img_size[0] * layer2_input_img_size[1],
                         n_out=100, activation=T.tanh, \
                         W=layer2_state[0], b=layer2_state[1] \
                         )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=100, n_out=N_OUT, \
                                    W=layer3_state[0], b=layer3_state[1] \
                                )

def train_by_lenet5(tr_start_index, tr_limit, vl_start_index, vl_limit, learning_rate=0.01, n_epochs=200):

    global train_dataset_route
    global valid_dataset_route

    print train_dataset_route, type(train_dataset_route)
    """
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    train_set = tdtf.read_data_patch_to_ndarray(train_dataset_route, tr_start_index, tr_limit)
    datasets = load_data.shared_dataset(train_set)
    train_set_x, train_set_y = datasets

    valid_set = tdtf.read_data_patch_to_ndarray(valid_dataset_route, vl_start_index, vl_limit)
    print valid_set[1]
    datasets = load_data.shared_dataset(valid_set)
    valid_set_x, valid_set_y = datasets

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size

    # allocate symbolic variables for the data

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    validate_model = theano.function([index], layer3.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    train_model = theano.function([index], [cost, layer3.errors(y), layer3.params[0][0][0]], updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 500 # look as this many examples regardless
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
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    min_train_cost = 10000
    decreasing_num = 0

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter , ' patience = ' , patience
            cost_ij, error_rate, par = train_model(minibatch_index)
            print('epoch %i, minibatch %i/%i, train cost %.2f , error rate %.2f %% , paramsW00 %f ' % \
                  (epoch, minibatch_index + 1, n_train_batches, \
                   cost_ij, error_rate*100., par))
            
            #print layer1.params[0:1][0][0:3]
            #print layer2.params[0:1][0][0:3]
            if cost_ij < min_train_cost:
                decreasing_num = 0
                min_train_cost = cost_ij
                '''
                layer0_state = layer0.__getstate__()
                layer1_state = layer1.__getstate__()
                layer2_state = layer2.__getstate__()
                layer3_state = layer3.__getstate__()
                trained_model_list = [layer0_state, layer1_state, layer2_state, layer3_state]
                trained_model_array = numpy.asarray(trained_model_list)
                classifier_file = open(train_model_route, 'w')
                cPickle.dump([1,2,3], classifier_file, protocol=2)
                numpy.save(classifier_file, trained_model_array)
                classifier_file.close()
                ''' 
            else:
                print "decreasing"
                decreasing_num += 1
                if decreasing_num > 100:
                    done_looping = True
                    break
            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
            if patience <= iter:
                done_looping = True
                print patience , iter
                break

    end_time = time.clock()
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))

    if_trained_yet = True

def run_train():
    if if_trained_yet:
        load_trained_model()
    train_by_lenet5( \
        tr_start_index=1, tr_limit=100 , \
        vl_start_index=1, vl_limit=100 \
    )

if __name__ == '__main__':
    start_sec = time.time()
    run_train()
    end_sec = time.time()
    print 'practical using time : %.2fm ' % ((end_sec - start_sec) / 60.)

def experiment(state, channel):
    train_by_lenet5(state.learning_rate, dataset=state.dataset)
