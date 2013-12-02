#!/usr/bin/python
'''
written by hp_carrot
2013-12-02
image recognition using logisticRegression
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

import load_data

DataHome = "../../data/Kaggle/DogVsCatData/"
ModelHome = "../trained_model/"
train_model_route = ModelHome + "DogVsCat_bghead2c_model_ls.np.pkl"

if_load_trained_model = False

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')   # the data is presented as rasterized images

layer0_input_shape = (50, 50)
layer0_output_shape = 2

in_shape = layer0_input_shape[0] * layer0_input_shape[1]
classifier = LogisticRegression(input=x, n_in=in_shape, n_out=layer0_output_shape)

# definition for theano.function
test_results = theano.function(inputs=[x], \
    outputs=classifier.y_pred)

def load_trained_model():
    global if_load_trained_model
    global train_model_route 
    global classifier
    global validate_model

    if_load_trained_model = True 
    print "loading trained model for the first time"
    trained_model_pkl = open(train_model_route, 'r')
    trained_model_state_list = cPickle.load(trained_model_pkl)
    trained_model_state_array = numpy.load(trained_model_pkl)
    classifier_state = trained_model_state_array[0]

    classifier = LogisticRegression(input=x, n_in=in_shape, n_out=layer0_output_shape
                                    , W=classifier_state[0], b=classifier_state[1])

    # definition for theano.function
    test_results = theano.function(inputs=[x], \
        outputs=classifier.y_pred)

def image_recognition(img):
    
    """ lenet
    :type img: ndarray
    :param img: img to identify

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    global if_load_trained_model
    global test_results

    if if_load_trained_model == 0:
        load_trained_model()

    start_time = time.clock()
    #convert to gray.  the result shape is (h,w)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # to resize 
    w,h = layer0_input_shape
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
    img_route = DataHome + "bg/cat.0.jpg"
    test_image_recgonition(img_route)
    img_route = DataHome + "bg/cat.5100.jpg"
    test_image_recgonition(img_route)
    img_route = DataHome + "bg/cat.5101.jpg"
    test_image_recgonition(img_route)
