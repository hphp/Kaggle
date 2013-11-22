#!/usr/bin/python
'''
written by hp_carrot
2013-11-22

2013-11-18
test with trained model
lenet
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
ModelHome = "../trained_model/"
train_model_route = ModelHome + "DogVsCat_trained_model_lenet_2500_feature.np.pkl"
test_label_route = DataHome + "DogVsCat_lenet_2500f.csv"

def evaluate_lenet5(dataset_route=DataHome+"DogVsCat_test_feature_2500.csv", \
                    nkerns=[20, 50], batch_size=5):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    trained_model_pkl = open(ModelHome + train_model_route, 'r')
    trained_model_state_list = cPickle.load(trained_model_pkl)
    trained_model_state_array = numpy.load(trained_model_pkl)
    layer0_state, layer1_state, layer2_state, layer3_state = trained_model_state_array

    test_set = tdtf.read_data_to_ndarray(dataset_route, limit=None, header_n=0)
    test_set_x, id_arr = test_set
    datasets = load_data.shared_dataset(test_set)
    test_set_x, test_set_y = datasets
    print test_set_x.shape, test_set_y.shape

    # compute number of minibatches for training, validation and testing
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ishape = (50, 50)  # this is the size of MNIST images

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 50, 50))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input, \
            image_shape=(batch_size, 1, 50, 50), \
            filter_shape=(nkerns[0], 1, 10, 10), poolsize=(2, 2), \
            W=layer0_state[0], b=layer0_state[1] \
            )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], 20, 20),
            filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2), \
            W=layer1_state[0], b=layer1_state[1] \
            )

    # the TanhLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * 8 * 8,
                         n_out=100, activation=T.tanh,\
                         W=layer2_state[0], b=layer2_state[1] \
                         )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=100, n_out=2, \
                                    W=layer3_state[0], b=layer3_state[1] \
                                )

    print "predicting"
    start_time = time.clock()
    # create a function to compute the mistakes that are made by the model
    test_results = theano.function(inputs=[index],
            outputs= layer3.y_pred,
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size]})

    
    test_res = [test_results(i)
        for i in xrange(n_test_batches)]
    print test_res
   
    id_l = []
    label_l = []
    index = 0
    for arr in test_res:
        for label in arr:
            label_l.append(label)
            id_l.append(id_arr[index])
            index += 1
    tdtf.wr_to_csv(header=['id','label'], id_list=id_l, pred_list=label_l, filename=test_label_route)
    end_time = time.clock()
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    start_sec = time.time()
    evaluate_lenet5()
    end_sec = time.time()
    print 'practical using time : %.2fm ' % ((end_sec - start_sec) / 60.)

def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
