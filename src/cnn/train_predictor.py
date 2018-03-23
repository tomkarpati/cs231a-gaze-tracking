# Run tensor flow

import sys
import os
sys.path.append(os.path.dirname(__file__)+"/cnn")

import argparse

import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the predictor classes
import pTfCNNPredictor as p
# Load the dataset class
import pData as d

print "Running "+str(sys.argv)

space = {}

parser = argparse.ArgumentParser(description="predictor option parser")
parser.add_argument('trainset', help="Training dataset name")
parser.add_argument('valset', help="Validation dataset name")
args = parser.parse_args()

image_space = {}
image_space['image_x'] = 32
image_space['image_y'] = 32
image_space['image_c'] = 4
space['image'] = image_space

space['num_conv_layers'] = 3

conv_layer_0_space = {}
conv_layer_0_space['filter'] = [ 3, 32 ]
conv_layer_0_space['stride'] = 1
conv_layer_0_space['pooling'] = [1, 1]
space['conv_layer_0'] = conv_layer_0_space

conv_layer_1_space = {}
conv_layer_1_space['filter'] = [ 5, 64 ]
conv_layer_1_space['stride'] = 1
conv_layer_1_space['pooling'] = [2, 2]
space['conv_layer_1'] = conv_layer_1_space

conv_layer_2_space = {}
conv_layer_2_space['filter'] = [ 5, 64 ]
conv_layer_2_space['stride'] = 1
conv_layer_2_space['pooling'] = [2, 2]
space['conv_layer_2'] = conv_layer_2_space

space['num_fc_layers'] = 3
space['fc_layer_0'] = 1024
space['fc_layer_1'] = 1024
space['fc_layer_2'] = 1024
space['readout_vec'] = 2

# Build the predictor
predictor = p.pTfCNNPredictor(space,
                              tensorboard=True,
                              logging=True,
                              verbose=True)

# Load the dataset
trainData = d.pData(verbose=True)
trainData.load_dataset(filename=args.trainset)
#valData = d.pData(verbose=True)
#valData.load_dataset(filename=args.valset)

# Collect some stats for the input.
print "image min: {}".format(np.min(trainData.eyeL))
print "image max: {}".format(np.max(trainData.eyeL))
print "image mean: {}".format(np.mean(trainData.eyeL))
print "target x min: {}".format(np.min(trainData.targetVec, axis=0))
print "target x max: {}".format(np.max(trainData.targetVec, axis=0))
print "target x mean: {}".format(np.mean(trainData.targetVec, axis=0))

# Train this thing
predictor.train(trainingSet=trainData,
                testSet=None,
                epochs=100)

predictor.save()
