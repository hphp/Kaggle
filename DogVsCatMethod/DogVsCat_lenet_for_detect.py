#!/usr/bin/python
'''
written by hp_carrot
2013-11-19
image recognition using lenet
'''

import cPickle
import os
import sys
import time

import numpy

import theano
theano.config.floatX='float32'
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from PIL import Image

if not "../DL_Method/" in sys.path:
    sys.path.append("../DL_Method/")
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from convolutional_mlp import LeNetConvPoolLayer

DataHome = "../../data/Kaggle/DogVsCatData/"
ModelHome = "../trained_model/"
train_model_route = ModelHome + "DogVsCat_trained_model_lenet_head_feature_2500.pkl"

def shared_dataset(data, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x = data
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x

def image_recognition(img_array):
    
    """ lenet
    :type img_array: ndarray
    :param img_array: img to identify

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    start_time = time.clock()

    img_array = numpy.asarray(img_array,dtype='float32') / 255.
    dataset = [img_array]
    print type(dataset)
    test_set_x = shared_dataset(dataset)
    print type(test_set_x),test_set_x.shape

    nkerns=[20, 50]
    rng = numpy.random.RandomState(23455)
    batch_size = 1

    trained_model_pkl = open(train_model_route, 'r')
    trained_model_state_list = cPickle.load(trained_model_pkl)
    layer0_state, layer1_state, layer2_state, layer3_state = trained_model_state_list 

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

    # Reshape matrix of rasterized images of shape (1, 50*50)
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

    test_results = theano.function(inputs=[index],
            outputs= layer3.y_pred,
            givens={
                x: test_set_x[index:index+1]})

    print "predicting"

    img_label = test_results(0)
    
    end_time = time.clock()
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %d secs' % ((end_time - start_time)))
    return img_label[0]

if __name__ == '__main__':
    img_route = DataHome + "head_images/keeshond_149.jpg"
    img_route = DataHome + "head_images/Abyssinian_100.jpg"
    img = Image.open(img_route)
    img = img.resize((50, 50), Image.ANTIALIAS)
    img = numpy.asarray(img, dtype='int32')
    n_img = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cur_v = 0
            for k in range(img.shape[2]):
                cur_v += img[i][j][k]
            cur_v /= 3
            n_img.append(cur_v)
    img = numpy.asarray(n_img)
    print type(img),img.size
    img_label = image_recognition(img)
    print img_label
