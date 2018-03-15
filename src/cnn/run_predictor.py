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
parser.add_argument('dataset', help="Dataset name")
args = parser.parse_args()

image_space = {}
image_space['image_x'] = 32
image_space['image_y'] = 32
image_space['image_c'] = 1
space['image'] = image_space

space['num_conv_layers'] = 3

conv_layer_0_space = {}
conv_layer_0_space['filter'] = [ 3, 16 ]
conv_layer_0_space['stride'] = 1
conv_layer_0_space['pooling'] = [1, 1]
space['conv_layer_0'] = conv_layer_0_space

conv_layer_1_space = {}
conv_layer_1_space['filter'] = [ 3, 64 ]
conv_layer_1_space['stride'] = 1
conv_layer_1_space['pooling'] = [2, 2]
space['conv_layer_1'] = conv_layer_1_space

conv_layer_2_space = {}
conv_layer_2_space['filter'] = [ 3, 64 ]
conv_layer_2_space['stride'] = 1
conv_layer_2_space['pooling'] = [1, 1]
space['conv_layer_2'] = conv_layer_2_space

space['num_fc_layers'] = 3
space['fc_layer_0'] = 1024
space['fc_layer_1'] = 1024
space['fc_layer_2'] = 1024
space['readout_x'] = 1

# Build the predictor
predictor = p.pTfCNNPredictor(space,
                              tensorboard=True,
                              logging=True,
                              verbose=True)

# Load the dataset
data = d.pData(verbose=True)
data.load_dataset(filename=args.dataset)

# Collect some stats for the input.
print "image min: {}".format(np.min(data.eyeL))
print "image max: {}".format(np.max(data.eyeL))
print "image mean: {}".format(np.mean(data.eyeL))
print "target x min: {}".format(np.min(data.targetX))
print "target x max: {}".format(np.max(data.targetX))
print "target x mean: {}".format(np.mean(data.targetX))

# Train this thing
predictor.train(trainingSet=data,
                testSet=None,
                epochs=100)
