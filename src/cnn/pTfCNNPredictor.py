# This is a multilayer nerural network

import sys
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class pTfCNNPredictor:
    # This is a subclass of the generic predictor

    def __init__(self,
                 space,
                 optType='adam',
                 tensorboard=False,
                 logging=False,
                 verbose=False):

        print "Initializing pTfNNPredictor";

        self.verbose = verbose
        self.tensorboard=tensorboard
        self.logging=logging

        self.learning_rate = 0.001

        self.logFH = None
        self.logLocation = "."
        
        # Build the model
        self.build_model(space)

        # Setup logging
        self.init_logging()

        # Setup the loss and build the optimizer
        (self.meanLoss, self.totalLoss) = self.loss_func(self.targetXClass,self.score)
        self.optimizer(optType=optType)

        self.init_session()
        

    # Build the model.
    # Everything is defined in the "space" dictionary
    def build_model(self, space):
        # Build the model configured to the space defined

        # Reset the graph so we can build multiple models in the
        # same python session
        tf.reset_default_graph()

        # Define the palceholders
        def placeholder_eye(space, name):
            print "Building image input {}".format(name)
            return tf.placeholder(tf.float32, [None,
                                             space['image_x'],
                                             space['image_y'],
                                             space['image_c']],
                                  name=name)

        self.eyeL = placeholder_eye(space['image'],"eye_l")
        #self.eyeR = placeholder_eye(space['image'],"eye_r")

        def placeholder_targetClass(name):
            # The class placeholders have batchx1 dimensions
            print "Building target {}".format(name)
            return tf.placeholder(tf.int32,
                                  [None],
                                  name=name)
        
        self.targetXClass = placeholder_targetClass("target_x")
        
        def placeholder_bb(name):
            print "Building bounding box input {}".format(name)
            return tf.placeholder(tf.float32, [None,
                                               4],
                                  name=name)
        
        #self.eyeLBB = placeholder_bb("eye_l_bb")
        #self.eyeRBB = placeholder_bb("eye_r_bb")
        #self.faceBB = placeholder_bb("face_bb")

        def add_tf_variable(name, space, inittype='normal'):
            if (inittype == 'normal'):
                initVal = 0.0
                print "Initialize variable {} with mean {}".format(name, initVal)
                init = tf.truncated_normal(shape=space, mean=initVal, stddev=0.1)
            elif (inittype == 'const'):
                initVal = 0.1
                print "Initialize variable {} with const {}".format(name, initVal)
                init = tf.constant(initVal, shape=space)
            else:
                assert(0)

            v = tf.Variable(init, name=name)
            return v

        def add_conv_layer(name, x, space):
            print "Building convolutional layer {}, with input {}, and space {}".format(name, x, space)
            with tf.name_scope(name):

                # Build a conv2 layer
                conv = tf.layers.conv2d(
                    inputs=x,
                    filters=space['filter'][1],
                    kernel_size=space['filter'][0],
                    padding="same",
                    activation=tf.nn.relu,
                    use_bias=True,
                    name=name)

                pool = tf.layers.max_pooling2d(inputs=conv,
                                               pool_size=space['pooling'][0],
                                               strides=space['pooling'][1])
                
                if name=="conv_layer_0":
                    print "storing {}".format(name)
                    self.tmp = conv

                
                tf.summary.histogram("conv",conv)
                tf.summary.histogram("pooling",pool)
                
                if self.verbose:
                    print conv
                    print pool

            return pool
                    

        def add_fc_layer(name, x, space):
            print "Building layer {}, with input {}, and space {}".format(name, x, space)
            with tf.name_scope(name):
                WSpace = [np.shape(x)[1].value, space]
                W = add_tf_variable("W",
                                    WSpace)
                b = add_tf_variable("b",
                                    [space],
                                    inittype="const")
                mult = tf.matmul(x,W)
                act = mult + b

                # Run through non-linear function
                h = tf.nn.relu(act)

                tf.summary.histogram("W",W)
                tf.summary.histogram("b",b)
                tf.summary.histogram("mult",mult)
                tf.summary.histogram("activation",act)
                tf.summary.histogram("hidden",h)
                
                if self.verbose:
                    print W
                    print b
                    print mult
                    print act
                    print h


            return (h,act)

        # Flatten out the images and turn into a single vector
        def vectorize(i):
            dim = np.prod(i.get_shape().as_list()[1:])
            return tf.reshape(i, [-1, dim])

        print self.eyeL
        print self.targetXClass


        # Add our convolutional layers
        x = self.eyeL
        for l in range(0,space['num_conv_layers']):
            name="conv_layer_"+str(l)
            x = add_conv_layer(name, x, space[name])
            
        # Turn this into a single vector
        x = vectorize(x)

        for l in range(0,space['num_fc_layers']):
            name="fc_layer_"+str(l)
            (x,act) = add_fc_layer(name, x, space[name])

        self.score = act

    
    def init_logging(self, location=None):
        if self.logging:
            if (location is not None):
                self.logLocation = location

            self.logFile = self.logLocation+"/stats.out"
            directory = os.path.dirname(self.logFile)
            if not os.path.exists(directory):
                os.makedirs(directory)

            self.logFH = open(self.logFile,'w')
            


    # Train the predictor
    def train(self,
              trainingSet,
              testSet,
              epochs=1,
              batchSize=16):

        # This is the list of operations we need to run
        opList = []
            
        # Define some variables
        with tf.name_scope("train"):
            predictedXClass = tf.argmax(self.score,1)
            self.train_correct = tf.reduce_sum(tf.cast(tf.equal(self.targetXClass,
                                                                tf.cast(predictedXClass,
                                                                        tf.int32)),
                                                       tf.int8))

            if self.tensorboard:
                tf.summary.scalar("training accuracy", self.train_correct)
                self.merged = tf.summary.merge_all()
                opList.append(self.merged)

        opList.extend([self.train_correct,
                       self.score,
                       predictedXClass,
                       self.meanLoss,
                       self.totalLoss,
                       self.tmp,
                       self.train_step])
        

        # Run training
        for i in range(epochs):
            sys.stdout.write("Training: epoch {}:\n".format(i))
            sys.stdout.flush()

            epochCorrect = 0
            epochLoss = 0
            for batchStart in range(0, trainingSet.numSamples, batchSize):
                sys.stdout.write(".")
                sys.stdout.flush()
                
                batchEnd = batchStart + batchSize
                if (batchEnd > trainingSet.numSamples):
                    batchEnd = trainingSet.numSamples

                # The input is arranged as (rows=samples x colums=features)
                eyeL = trainingSet.eyeL[batchStart:batchEnd,:]
                targetXClass = trainingSet.targetXClass[batchStart:batchEnd]/2
                result = self.session.run(opList,
                                          feed_dict={self.eyeL : eyeL,
                                                     self.targetXClass : targetXClass})
                if self.tensorboard:
                    summary = result[0]
                    result = result[1:]
                    self.trainWriter.add_summary(summary,(i*trainingSet.numSamples) + batchStart)

                    
                (correct, score, predX, meanLoss, totalLoss, tmp, train) = result
                epochCorrect += correct
                epochLoss += totalLoss

                if self.verbose:
                    print "Training batch @{}: correct {}".format(batchStart,
                                                                  correct)
                    #print "Score: {}".format(score)
                    print "Loss: {}".format(meanLoss)
                    print "Predicted: {}".format(predX)
                    print "Label: {}".format(targetXClass)

            print "Total loss: {}".format(epochLoss*1.0/trainingSet.numSamples)
            print "Overall prediction rate: {}".format(epochCorrect*1.0/trainingSet.numSamples)


        #print "VAR: ({}) {}".format(np.shape(tmp),tmp)
        for im in range(0,np.size(tmp,axis=3)):
            #print "filtered im {}".format(im)
            fig = plt.figure()
            i = tmp[-1,:,:,im]
            np.shape(i)
            plt.imshow(tmp[-1,:,:,im],cmap="gray")
            plt.show()


            
    def predict(self):
        pass


    def loss_func(self, target, score):

        # Define a sub-loss for each directory(x or y)
        def sub_loss(target, score, name):
            # Perform softmax entropy at last layer
            with tf.name_scope(name):
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=target,
                    logits=score,
                    name="cross_entropy")
                mean = tf.reduce_mean(cross_entropy, name="mean_cross_entropy")
                total = tf.reduce_sum(cross_entropy, name="sum_cross_entropy")
                tf.summary.histogram("cross_entropy", cross_entropy)
                tf.summary.scalar("mean_cross_entropy", mean)
                return (mean, total)

        # Define the loss in X-direction
        (x_mean_loss, x_total_loss) = sub_loss(target, score, "loss_x")
        
        return (x_mean_loss, x_total_loss)
    

    def optimizer(self,
                  optType='gradient',
                  learningRate=0.01):
        # Separate out the training into it's own scope
        with tf.name_scope("train"):
            if optType=='gradient':
                print "Using GradientDecent Optimizer"
                self.train_step = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(self.meanLoss)
            elif optType=='adam':
                print "Using Adam Optimizer"
                self.train_step = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(self.meanLoss)
            else:
                assert(0)
                
                

    def init_session(self):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        if self.tensorboard:
            self.trainWriter = tf.summary.FileWriter(self.logLocation+"/tensorboard/train",
                                                     self.session.graph)
            


