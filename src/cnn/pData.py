# This is a data class for CNN training


import numpy as np
import cv2

class pData:
    # This holds the data from the training/test set

    def __init__(self,
                 imSize=[32,32, 1],
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
        dim = []
        dim.append(videoFrames)
        dim.extend(self.imSize)
        size = tuple(self.imSize[0:2])
        print "Video dimensions: {} , {}".format(dim,size)
        self.dataFrames = np.empty(dim)
        for i in range(0,videoFrames):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.dataFrames[i] = np.reshape(cv2.resize(gray, size), self.imSize)

            cv2.imshow('frame',gray)
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

        dataset = np.hstack([flattened, self.target])
        dataFile = directory+"/"+basename+".txt"

        np.savetxt(dataFile, dataset, delimiter=',', fmt='%1d')
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
        
        # We need to bin the target data
        def binTarget(a):
            amin = a.min()
            amax = a.max()
            if self.verbose:
                print "Binned data range is {},{}".format(amin, amax)
            r = amax - amin
            bs = r * 1.0/self.targetVectorClassSize
            logits = np.empty([len(a),self.targetVectorClassSize],
                              dtype=int)
            labels = np.empty([len(a)],
                              dtype=int)
            for i in range(len(a)):
                for j in range(self.targetVectorClassSize):
                    if (a[i] <= amin + (j+1)*bs):
                        labels[i] = j
                        logits[i,j] = 1
                        break
            return (logits, labels)
                        
        (self.targetX, self.targetXClass) = binTarget(d[:,np.prod(self.imSize)])
        if self.verbose:
            print "Left eye dataset size: {}".format(np.shape(self.eyeL))
            print "Target X dataset size (logits): {}".format(np.shape(self.targetX))
            print "Target X dataset size (labels): {}".format(np.shape(self.targetXClass))

        return
    
