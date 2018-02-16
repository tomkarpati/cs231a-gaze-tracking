#!/bin/sh

# Setup the environment

system=`uname -s`
pythonver=`python -c "import platform ; print platform.python_version()"`

# We use python 2.7

# Make sure that we have pip and virtual env
pip --version || (echo "pip not found" && exit)
virtualenv --version || (echo "virtualenv not found" && exit)


# Generate the virtualenv tree
virtualenv --system-site-packages cs231a-${system}-python_${pythonver}-env

