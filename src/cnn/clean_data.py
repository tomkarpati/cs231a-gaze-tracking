# Clean the data

import sys

import argparse

import pData

print "Running ",str(sys.argv)

parser = argparse.ArgumentParser(description="Data cleaning option parser")
parser.add_argument('--dataDir', default=".", help="Location of the dataset")
parser.add_argument('dataset', help="Dataset name")
args = parser.parse_args()

print "Data directory: {}".format(args.dataDir)
print "Dataset: {}".format(args.dataset)

d = pData.pData(verbose=True)

d.read_data(args.dataset,
            args.dataDir)


