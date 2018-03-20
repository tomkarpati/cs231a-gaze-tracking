# Clean the data

import sys

import argparse
import glob

import pData

print "Running ",str(sys.argv)

parser = argparse.ArgumentParser(description="Data cleaning option parser")
parser.add_argument('--dataDir', default=".", help="Location of the dataset")
args = parser.parse_args()

print "Data directory: {}".format(args.dataDir)

d = pData.pData(verbose=True)

# Grab all of the videos from the directory
for s in glob.iglob(args.dataDir+"/*.avi"):
    basename = s[:-4] # strip off the suffix
    l = basename.rsplit("/",1)
    dataset = l[-1]
    print "Dataset: {}".format(dataset)
    d.read_data(dataset,
                args.dataDir)
    
    #d.write_dataset(dataset)
