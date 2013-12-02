#!/usr/bin/python
'''
written by hp_carrot
2013-11-30
image recognition using logistic regression
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
if not "../DataProcess/" in sys.path:
    sys.path.append("../DataProcess/")
import transform_data_to_format as tdtf

import load_data

DataHome = "../../data/Kaggle/DogVsCatData/"
ModelHome = DataHome #"../trained_model/"
train_model_route = ModelHome + "DogVsCat_bghead2c_model_ls.np.pkl"
valid_dataset_route = DataHome + "DogVsCat_bghead_train_feature_2c_2500.csv"

if_load_trained_model = 0

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')   # the data is presented as rasterized images
y = T.ivector('y')   # the data is presented as rasterized images

layer0_input_shape = (50, 50)
layer0_output_shape = 2

in_shape = layer0_input_shape[0] * layer0_input_shape[1]
classifier = LogisticRegression(input=x, n_in=in_shape, n_out=layer0_output_shape)

# definition for theano.function
validate_model = theano.function(inputs=[x, y], \
                    outputs=classifier.errors(y))

def load_trained_model():
    global if_load_trained_model
    global train_model_route 
    global classifier
    global validate_model

    if_load_trained_model = 1
    print "loading trained model for the first time"
    trained_model_pkl = open(train_model_route, 'r')
    trained_model_state_list = cPickle.load(trained_model_pkl)
    trained_model_state_array = numpy.load(trained_model_pkl)
    classifier_state = trained_model_state_array[0]

    classifier = LogisticRegression(input=x, n_in=in_shape, n_out=layer0_output_shape
                                    , W=classifier_state[0], b=classifier_state[1])

    # definition for theano.function
    validate_model = theano.function(inputs=[x, y],
            outputs=classifier.errors(y))

def validation_err(vl_start_index=0, vl_limit=None):
    
    global if_load_trained_model
    global validate_model

    if if_load_trained_model == 0:
        load_trained_model()
    
    valid_set = tdtf.read_data_patch_to_ndarray(valid_dataset_route, vl_start_index, vl_limit)
    print valid_set[1]
    valid_set_x, valid_set_y = valid_set
    validation_loss = validate_model(valid_set_x, valid_set_y)
    print('validation error %f %%' % \
        (validation_loss * 100.))


if __name__ == '__main__':
    validation_err()
