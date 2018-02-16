#!/bin/sh

if [ ! $VIRTUAL_ENV ]; then
    echo "*** ERROR : This ($0) needs to run in a virtual environment ***"
    exit
fi

# Get the top level
directory=$(cd `dirname $0` && cd .. && pwd)
echo "Top level: "$directory

opencv_git_repo=$directory/submodules/opencv
python_site_packages=`python -m site | grep "$VIRTUAL_ENV.*site-packages" | sed "s/.*\'\(.*\)\',/\1/"`


echo "Python site packages: "$python_site_packages

opencv_lib=$directory/lib/opencv
opencv_build=$opencv_git_repo/build


if [ ! -d "$opencv_git_repo" ]; then
    echo "OpenCV repo doesn't exist."
    exit -1
fi

if [ ! -d "$opencv_lib" ]; then
    mkdir -pv $opencv_lib
fi
if [ ! -d "$opencv_build" ]; then
    mkdir -pv $opencv_build
fi

pushd $opencv_build

pwd

cmake -DCMAKE_BUILD_TYPE=Release -DBUILD-DOCS=ON -DCMAKE_INSTALL_PREFIX=$opencv_lib -DPYTHON2_PACKAGES_PATH=$python_site_packages $opencv_git_repo | tee opencv.build.log 2>&1

make -j4

make install


# Run a simple test to see if it's loaded and the version number
popd

python src/opencv-test.py
