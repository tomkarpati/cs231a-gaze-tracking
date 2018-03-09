#!/bin/sh

# This should run in a vritual env.
# Otherwise crap out

if [ ! $VIRTUAL_ENV ]; then
    echo "*** ERROR : This ($0) needs to run in a virtual environment ***"
    exit
fi

echo "Checking virtual evironment"

# Install and update the necessary packages
pip install -U pip

# Install numpy
pip install --upgrade numpy

# Install opencv
pip install --upgrade opencv-python

# Install tensor flow
pip install --upgrade tensorflow
