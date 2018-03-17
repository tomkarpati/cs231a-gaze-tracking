# This is a data class for CNN training


import numpy as np
import cv2
import matplotlib.pyplot as plt

class pData:
    # This holds the data from the training/test set

    def __init__(self,
                 imSize=[32, 32, 4],
                 verbose=False):
        self.verbose = verbose
        self.imSize = imSize
        self.targetVectorClassSize = 50

        self.dataFrames = None
        
        print "Current status:"
        print "-- verbose: {}".format(verbose)
        print "-- image size: {}".format(imSize)


    # We can concatinate multiple of these calls together
    def read_data(self,
                  basename,
                  directory="."):

        # We can concatinate multiple data files together
        # basename is the name of the testset with directory

        # Get the AVI file
        aviFile = directory+"/"+basename+".avi"
        # Get the mapping file
        mapFile = directory+"/"+basename+".dat"
        print "Opening AVI file: {}".format(aviFile)

        # Open the file.
        cap = cv2.VideoCapture(aviFile)

        if ( not cap.isOpened() ): exit(-1)
        
        # Create the storage to load the images
        videoFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        videoWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        videoHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print "Video contains {} frames of {}x{} size".format(
            videoFrames,
            videoWidth,
            videoHeight)
        print "Rescaling to {}".format(self.imSize)

        self.numSamples = self
        

        # Read the data into an array
        print "Reading video..."
        frameDim = tuple(self.imSize[0:2])
        videoDim = [videoFrames]
        videoDim.extend(self.imSize)
        print "Video dimensions: {}".format(videoDim)
        self.dataFrames = np.empty(videoDim)
        for i in range(0,videoFrames):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.dataFrames[i,:,:,0:3] = cv2.resize(frame, frameDim)*1.0/255
            pupils = np.zeros(np.shape(gray))

            # Attempt to highlight the pupils in the image
            circles = cv2.HoughCircles(gray,
                                       cv2.HOUGH_GRADIENT,
                                       dp=128,
                                       minDist=2000,
                                       minRadius=50,
                                       maxRadius=150)
            if (circles is not None):
                print "Frame {}, Circles: {}".format(i, len(circles))
                for c in circles[0]:
                    print "circle: {}".format(c)
                    cv2.circle(frame, (c[0],c[1]), c[2], (0,255,0),  5)

                    # Write an alpha channel with the pupils
                    cv2.circle(pupils, (c[0],c[1]), c[2], (255,255,255), -1)
                    
                #self.pupils[i] = np.reshape(cv2.resize(frame, size), self.imSize)
                
            # Resize the pupil locations and write to alpha channel
            self.dataFrames[i,:,:,3] = cv2.resize(pupils, frameDim)*1.0/255
                
            cv2.imshow("Frame",frame)
            cv2.waitKey(10)

            
        print "Done."

        print "Reading homography matricies..."
        # Read the homography data into arrays
        d = np.loadtxt(mapFile,delimiter=',')
        if self.verbose:
            print d
            print np.shape(d)
        assert (np.shape(d) == (videoFrames, 34))
        self.H_eyes = np.reshape(d[:,0:16], [-1, 4, 4])
        self.H_monitor = np.reshape(d[:,16:32], [-1, 4, 4])
        self.target = d[:,32:34]

        print "Sanity checking..."
        assert (np.size(self.H_eyes,axis=0) == videoFrames)
        assert (np.size(self.H_monitor,axis=0) == videoFrames)
        if self.verbose:
            print "H_eyes:\n{}".format(self.H_eyes)
            print "H_monitor:\n{}".format(self.H_monitor)
            print "Target:\n{}".format(self.target)
        print "Done."

        print "Dataset statistics:"
        print "-- Target X range: ({}, {})".format(np.min(self.target[:,0]),
                                                   np.max(self.target[:,0]))
        print "-- Target Y range: ({}, {})".format(np.min(self.target[:,1]),
                                                   np.max(self.target[:,1]))
        print "Done."

        return

    def write_dataset(self,
                      basename,
                      directory="."):
        
        print "Flattening and writing dataset..."

        # Flatten the image to a vector. One row per frame
        # Append the target to the end of the row
        flattened = np.reshape(self.dataFrames, [-1, np.prod(self.imSize)])
        print np.shape(flattened)

        dataset = np.hstack([flattened, self.target])
        dataFile = directory+"/"+basename+".txt"

        np.savetxt(dataFile, dataset, delimiter=',')
        print "Done."

        
        return
        
    def load_dataset(self,
                     filename):
        
        print "Loading dataset from file..."
        d = np.loadtxt(filename,
                       delimiter=",")
        print "Done."

        # Shuffle the rows
        np.random.shuffle(d)
        
        print "Arranging data..."
        self.numSamples = len(d)
        
        eyeLSize = [-1]
        eyeLSize.extend(self.imSize)
        eyeLData = d[:,0:np.prod(self.imSize)]
        # reshape 
        self.eyeL = np.reshape(eyeLData, eyeLSize)
        self.eyeR = self.eyeL
        self.eyeLBB = np.tile([0, 0, 32, 32],(self.numSamples,1))
        self.eyeRBB = np.tile([0, 0, 32, 32],(self.numSamples,1))
        self.faceBB = np.tile([0, 0, 32, 32],(self.numSamples,1))
        print np.shape(self.eyeL)
        print np.shape(self.eyeR)
        print np.shape(self.eyeLBB)
        print np.shape(self.eyeRBB)
        print np.shape(self.faceBB)

        vecStart = np.prod(self.imSize)
        
        self.targetVec = d[:,vecStart:vecStart+2]

        print self.targetVec
        minTargetVec = np.min(self.targetVec,axis=0)
        maxTargetVec = np.max(self.targetVec,axis=0)

        self.targetVec = self.targetVec - (maxTargetVec-minTargetVec)/2
        self.targetVec = self.targetVec/(maxTargetVec - minTargetVec)


        print self.targetVec
        
        return
    
