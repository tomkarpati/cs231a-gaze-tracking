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
        (self.meanLoss, self.totalLoss) = self.loss_func(self.targetVec,self.scoreVec)
        self.optimizer(optType=optType)

        self.saver = tf.train.Saver()
        
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
        self.eyeR = placeholder_eye(space['image'],"eye_r")

        def placeholder_targetVec(name):
            # The class placeholders have batchx1 dimensions
            print "Building target (vector) {}".format(name)
            return tf.placeholder(tf.float32,
                                  [None, 2],
                                  name=name)
        
        self.targetVec = placeholder_targetVec("targetVec")
        
        def placeholder_bb(name):
            print "Building bounding box input {}".format(name)
            return tf.placeholder(tf.float32, [None,
                                               4],
                                  name=name)
        
        self.eyeLBB = placeholder_bb("eye_l_bb")
        self.eyeRBB = placeholder_bb("eye_r_bb")
        self.faceBB = placeholder_bb("face_bb")

        print self.eyeL
        print self.eyeR
        print self.targetVec
        print self.eyeLBB
        print self.eyeRBB
        print self.faceBB

        
        # Define some functions to create our layers
        def add_conv_layer(name, x, space):
            print "Building convolutional layer {}, with input {}, and space {}".format(name, x, space)
            # Build a conv2 layer
            with tf.name_scope(name):
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
                                               strides=space['pooling'][1],
                                               name=name)
                
                if name=="conv_layer_0_l":
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
            # Build a fully connected layer
            with tf.name_scope(name):
                h = tf.layers.dense(inputs=x,
                                    units=space,
                                    activation=tf.nn.relu,
                                    use_bias=True,
                                    name=name)
            
                tf.summary.histogram("hidden",h)
            
                if self.verbose:
                    print h


            return h

        def add_readout_layer(name, x, space):
            print "Building readout layer {}, with input {}, and space {}".format(name,x,space)
            # Build a readout layer at the end
            # This will use a linear activation function
            with tf.name_scope(name):
                score = tf.layers.dense(inputs=x,
                                        units=space,
                                        use_bias=True,
                                        name=name)

                tf.summary.histogram("score",score)

                if self.verbose:
                    print score
                    
            return score

        
        # Flatten out the images and turn into a single vector
        def vectorize(i):
            dim = np.prod(i.get_shape().as_list()[1:])
            return tf.reshape(i, [-1, dim])

        cl = self.eyeL
        cr = self.eyeR
        

        # Add our convolutional layers
        for l in range(0,space['num_conv_layers']):
            name="conv_layer_"+str(l)
            cl = add_conv_layer(name+"_l", cl, space[name])
            cr = add_conv_layer(name+"_r", cr, space[name])
            
        # Turn this into a single vector
        xl = vectorize(cl)
        xr = vectorize(cr)

        # Merge the two flattened convolution outputs
        x = tf.concat([xl,
                       xr,
                       self.eyeLBB,
                       self.eyeRBB,
                       self.faceBB], axis=1)
        
        for l in range(0,space['num_fc_layers']):
            name="fc_layer_"+str(l)
            x = add_fc_layer(name, x, space[name])

        name = "readout_vec"
        self.scoreVec = add_readout_layer(name, x, space[name])

    
    def init_logging(self, location=None):
        if self.logging:
            if (location is not None):
                self.logLocation = location

            self.logFile = self.logLocation+"/stats.out"
            directory = os.path.dirname(self.logFile)
            if not os.path.exists(directory):
                os.makedirs(directory)

            self.logFH = open(self.logFile,'w')
            

    def loss_func(self, target, score):

        print target
        print score
        
        # Define a sub-loss for each segment type
        def sub_loss(target, score, name):
            # Compute the error
            with tf.name_scope(name):
                se = (tf.norm((target - score),axis=1))**2
                print se
                mse = tf.reduce_mean(se, name="mean_error")
                print mse
                tse = tf.reduce_sum(se, name="sum_error")
                print tse
                tf.summary.histogram("error", se)
                tf.summary.scalar("mean_error", mse)

                self.se = se
                
                return (mse, tse)
            
        # Define the loss in X-direction
        (dir_mean_loss, dir_total_loss) = sub_loss(target, score, "loss")

        return (dir_mean_loss, dir_total_loss)

    # Train on-line
    def train_on_line(self,
                      counter,
                      eyeL,
                      eyeR,
                      eyeLBB,
                      eyeRBB,
                      faceBB,
                      target):
        
        opList = [self.meanLoss,
                  self.scoreVec,
                  self.se,
                  self.tmp]
        trainOpList = []

        if self.tensorboard:
            # Define some variables
            with tf.name_scope("train"):
                tf.summary.scalar("Training mean loss", self.meanLoss)
            
            self.merged = tf.summary.merge_all()
            trainOpList.append(self.merged)

        trainOpList.extend(opList)
        trainOpList.append(self.train_step)

        # Run training
        sys.stdout.write("Training ({}) - target: {}".format(counter, target))
        sys.stdout.flush()

        if self.logging:
            s = "Training ({}): eyeLBB: {}, eyeRBB: {}, faceBB: {}".format(counter, eyeLBB, eyeRBB, faceBB)
            self.logFH.write(s)

            # Reshape to match None as first dimension
            result = self.session.run(trainOpList,
                                      feed_dict={self.eyeL : eyeL,
                                                 self.eyeR : eyeR,
                                                 self.eyeLBB : eyeLBB,
                                                 self.eyeRBB : eyeRBB,
                                                 self.faceBB : faceBB,
                                                 self.targetVec : target})
            if self.tensorboard:
                summary = result[0]
                result = result[1:]
                self.trainWriter.add_summary(summary)

                    
            (meanLoss, predVec, se, tmp, train) = result
            
            if self.logging:
                s  = str(meanLoss)+","
                s += str(se)+","
                s += "\n{},".format(predVec.T)
                s += "\n{},".format(target.T)
                s += "\n"
                self.logFH.write(s)

            if self.verbose:
                #print "Training batch @{}: correct {}".format(batchStart,
                #                                              num_correct)
                #print "Loss: {}".format(meanLoss)
                #print "Accuracy: {}".format(accuracy)
                pass
            else:
                sys.stdout.write(".")
                sys.stdout.flush()

        return predVec
                
                    
    # Train the predictor
    def train(self,
              trainingSet,
              testSet,
              epochs=1,
              batchSize=16):

        # This is the list of operations we need to run
        opList = [self.meanLoss,
                  self.totalLoss,
                  self.scoreVec,
                  self.se,
                  self.tmp]
        
        trainOpList = []
        # Define some variables
        with tf.name_scope("train"):
            tf.summary.scalar("Training batch mean loss", self.meanLoss)
            
            self.merged = tf.summary.merge_all()
            trainOpList.append(self.merged)

        trainOpList.extend(opList)
        trainOpList.append(self.train_step)

        testOpList = []
        # Define some variables
        with tf.name_scope("test"):
            tf.summary.scalar("Testing epock mean loss", self.meanLoss)
            
            self.merged = tf.summary.merge_all()
            testOpList.append(self.merged)
            
        testOpList.extend(opList)

            
        # Run training
        for i in range(epochs):
            sys.stdout.write("Training: epoch {}:\n".format(i))
            sys.stdout.flush()

            if self.logging:
                self.logFH.write("Training:\n")
                                 
            epochCorrect = 0
            epochLoss = 0
            for batchStart in range(0, trainingSet.numSamples, batchSize):
                
                batchEnd = batchStart + batchSize
                if (batchEnd > trainingSet.numSamples):
                    batchEnd = trainingSet.numSamples

                # The input is arranged as (axis0=samples)
                eyeL = trainingSet.eyeL[batchStart:batchEnd,:,:,0:4]
                eyeR = trainingSet.eyeR[batchStart:batchEnd,:,:,0:4]
                eyeLBB = trainingSet.eyeLBB[batchStart:batchEnd,:]
                eyeRBB = trainingSet.eyeRBB[batchStart:batchEnd,:]
                faceBB = trainingSet.faceBB[batchStart:batchEnd,:]
                # Reshape this into a column vector
                target = trainingSet.targetVec[batchStart:batchEnd]
                result = self.session.run(trainOpList,
                                          feed_dict={self.eyeL : eyeL,
                                                     self.eyeR : eyeR,
                                                     self.eyeLBB : eyeLBB,
                                                     self.eyeRBB : eyeRBB,
                                                     self.faceBB : faceBB,
                                                     self.targetVec : target})
                if self.tensorboard:
                    summary = result[0]
                    result = result[1:]
                    self.trainWriter.add_summary(summary,(i*trainingSet.numSamples) + batchStart)

                    
                (meanLoss, totalLoss, predVec, se, tmp, train) = result
                epochLoss += totalLoss

                #if (not self.tmp == None):
                #print self.tmp[0,:,:,0]
                #fig = plt.figure()
                #plt.imshow(self.tmp[0,:,:,0])
                #plt.show()
                
                if self.logging:
                    s  = str(i)+","
                    s += str(batchStart)+","
                    s += str(meanLoss)+","
                    s += str(se)+","
                    s += "\n{},".format(predVec.T)
                    s += "\n{},".format(target.T)
                    s += "\n"
                    self.logFH.write(s)

                if self.verbose:
                    #print "Training batch @{}: correct {}".format(batchStart,
                    #                                              num_correct)
                    #print "Loss: {}".format(meanLoss)
                    #print "Accuracy: {}".format(accuracy)
                    pass
                else:
                    sys.stdout.write(".")
                    sys.stdout.flush()


            print "Epoch mean loss: {}".format(epochLoss*1.0/trainingSet.numSamples)

            if (testSet is not None):
                # For each epoch, run forward inference
                eyeL = testSet.eyeL[:,:,:,0:4]
                target = testSet.targetVec[:]
                result = self.session.run(testOpList,
                                          feed_dict={self.eyeL : eyeL,
                                                     self.targetVec : target})
                if self.tensorboard:
                    summary = result[0]
                    result = result[1:]
                    self.trainWriter.add_summary(summary,i)

                (meanLoss, totalLoss, predVec, se, tmp) = result
                
                if self.logging:
                    s  = str(i)+","
                    s += str(batchStart)+","
                    s += str(meanLoss)+","
                    s += str(se)+","
                    s += "\n{},".format(predVec.T)
                    s += "\n{},".format(target.T)
                    s += "\n"
                    self.logFH.write(s)
                
                print "Mean loss: {}".format(meanLoss)
            
        #print "VAR: ({}) {}".format(np.shape(tmp),tmp)
        #for im in range(0,np.size(tmp,axis=3)):
        #    #print "filtered im {}".format(im)
        #    fig = plt.figure()
        #    i = tmp[-1,:,:,im]
        #    np.shape(i)
        #    plt.imshow(tmp[-1,:,:,im],cmap="gray")
        #    plt.show()


            
    def predict(self,
                counter,
                eyeL,
                eyeR,
                eyeLBB,
                eyeRBB,
                faceBB,
                target):
        
        opList = [self.meanLoss,
                  self.scoreVec,
                  self.se,
                  self.tmp]
        predictOpList = []

        if self.tensorboard:
            # Define some variables
            with tf.name_scope("infer"):
                tf.summary.scalar("Inference mean loss", self.meanLoss)
            
            self.merged = tf.summary.merge_all()
            predictOpList.append(self.merged)

        predictOpList.extend(opList)

        # Run training
        sys.stdout.write("Testing ({}) - target: {}".format(counter, target))
        sys.stdout.flush()

        if self.logging:
            s = "Testing ({}): eyeLBB: {}, eyeRBB: {}, faceBB: {}".format(counter, eyeLBB, eyeRBB, faceBB)
            self.logFH.write(s)

            # Reshape to match None as first dimension
            result = self.session.run(predictOpList,
                                      feed_dict={self.eyeL : eyeL,
                                                 self.eyeR : eyeR,
                                                 self.eyeLBB : eyeLBB,
                                                 self.eyeRBB : eyeRBB,
                                                 self.faceBB : faceBB,
                                                 self.targetVec : target})
            if self.tensorboard:
                summary = result[0]
                result = result[1:]
                self.trainWriter.add_summary(summary)

                    
            (meanLoss, predVec, se, tmp) = result
            
            if self.logging:
                s  = str(meanLoss)+","
                s += str(se)+","
                s += "\n{},".format(predVec.T)
                s += "\n{},".format(target.T)
                s += "\n"
                self.logFH.write(s)

            if self.verbose:
                #print "Testing batch @{}: correct {}".format(batchStart,
                #                                              num_correct)
                #print "Loss: {}".format(meanLoss)
                #print "Accuracy: {}".format(accuracy)
                pass
            else:
                sys.stdout.write(".")
                sys.stdout.flush()

        return predVec
                

    def save(self):
        path = "./model.ckpt"
        save_path = self.saver.save(self.session, path)
        print("Model saved in path: %s" % save_path)

    def load(self):
        # Restore variables from disk.
        path = "./model.ckpt"
        self.saver.restore(self.session, path)
        print("Model loaded from path: %s" % path)

        
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
            


