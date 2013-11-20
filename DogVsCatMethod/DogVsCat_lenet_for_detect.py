#!/usr/bin/python
'''
written by hp_carrot
2013-11-19
image recognition using lenet
'''

import cPickle
import cv2
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
train_model_route = ModelHome + "DogVsCat_trained_model_lenet_head_feature_3c_2500_bg.np.pkl"

if_load_trained_model = 0

rng = numpy.random.RandomState(23455)
nkerns=[20, 50]
batch_size = 1

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')   # the data is presented as rasterized images

# Reshape matrix of rasterized images of shape (1, 50*50)
# to a 4D tensor, compatible with our LeNetConvPoolLayer
layer0_input = x.reshape((batch_size, 1, 50, 50))

# Construct the first convolutional pooling layer:
# filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
# maxpooling reduces this further to (24/2,24/2) = (12,12)
# 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
layer0 = LeNetConvPoolLayer(rng, input=layer0_input, \
        image_shape=(batch_size, 1, 50, 50), \
        filter_shape=(nkerns[0], 1, 10, 10), poolsize=(2, 2) \
        )

# Construct the second convolutional pooling layer
# filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
# maxpooling reduces this further to (8/2,8/2) = (4,4)
# 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
        image_shape=(batch_size, nkerns[0], 20, 20),
        filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2) \
        )

# the TanhLayer being fully-connected, it operates on 2D matrices of
# shape (batch_size,num_pixels) (i.e matrix of rasterized images).
# This will generate a matrix of shape (20,32*4*4) = (20,512)
layer2_input = layer1.output.flatten(2)

# construct a fully-connected sigmoidal layer
layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * 8 * 8,
                     n_out=100, activation=T.tanh \
                     )

# classify the values of the fully-connected sigmoidal layer
layer3 = LogisticRegression(input=layer2.output, n_in=100, n_out=3 \
                            )

# definition for theano.function
test_results = theano.function(inputs=[x], \
    outputs= layer3.y_pred)

def load_trained_model():
    global if_load_trained_model
    global train_model_route 
    global layer0_input 
    global layer0 
    global layer1 
    global layer2_input 
    global layer2 
    global layer3 
    global test_results

    if_load_trained_model = 1
    print "loading trained model for the first time"
    trained_model_pkl = open(train_model_route, 'r')
    trained_model_state_list = cPickle.load(trained_model_pkl)
    trained_model_state_array = numpy.load(trained_model_pkl)
    layer0_state, layer1_state, layer2_state, layer3_state = trained_model_state_array

    ishape = (50, 50)  # this is the size of MNIST images

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

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

    layer2_input = layer1.output.flatten(2)
    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * 8 * 8,
                         n_out=100, activation=T.tanh,\
                         W=layer2_state[0], b=layer2_state[1] \
                         )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=100, n_out=3, \
                                    W=layer3_state[0], b=layer3_state[1] \
                                )
    test_results = theano.function(inputs=[x], \
        outputs= layer3.y_pred)


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

def image_recognition(img):
    
    """ lenet
    :type img: ndarray
    :param img: img to identify

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    global if_load_trained_model
    global test_set_x
    global test_results

    if if_load_trained_model == 0:
        load_trained_model()

    start_time = time.clock()
    #convert to gray.  the result shape is (h,w)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # to resize 
    w,h=(50,50)
    img = cv2.resize(img,(w,h),interpolation=cv2.INTER_LINEAR)
    #print "after resize and gray:",type(img),img.shape,img.dtype

    #show the gray img
    #cv2.imshow("w2",img)
    #cv2.waitKey(0)

    #reshape (h,w) to (h*w,) 
    img=img.reshape(w*h) 
    #print "after reshape::",type(img),img.shape,img.dtype

    img = numpy.asarray(img,dtype='float32') / 256.
    dataset = [img]
    #print type(dataset)

    #print "predicting"

    img_label = test_results(dataset[0:1])
    
    end_time = time.clock()
    '''
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %d secs' % ((end_time - start_time)))
    '''
    return img_label[0]

def test_image_recgonition(img_route):
    img = cv2.imread(img_route)
    print "orignal img: ",type(img),img.shape,img.dtype
    print img.size

    # show
    cv2.imshow("w1",img)
    cv2.waitKey(0)

    # do recognition
    img_label = image_recognition(img)
    print img_label

if __name__ == '__main__':
    img_route = DataHome + "train/dog.2.jpg"
    test_image_recgonition(img_route)
    img_route = DataHome + "head_images/keeshond_149.jpg"
    test_image_recgonition(img_route)
    img_route = DataHome + "head_images/Abyssinian_100.jpg"
    test_image_recgonition(img_route)
    img_route = DataHome + "head_images/Bengal_10.jpg"
    test_image_recgonition(img_route)
    img_route = DataHome + "head_images/basset_hound_10.jpg"
    test_image_recgonition(img_route)
    img_route = DataHome + "head_images/basset_hound_100.jpg"
    test_image_recgonition(img_route)
