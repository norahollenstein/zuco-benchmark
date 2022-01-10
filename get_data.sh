#!/bin/bash

#Install osf-client

echo "Installing osf-client"
pip install osfclient

# Get the data from osf
echo "Creating data dir"
mkdir data

# Get the test-data from osf This may take a while
echo "Downloading test-data from osf, this may take a while"
osf -p d7frw clone
echo "Finished downloading test-data now cleaning"
mv d7frw/dropbox data/test
rm -r d7frw

echo "Downloading NR train-data from osf, this may take a while"
osf -p 2abup clone
echo "Finished downloading NR train-data now cleaning"
mv 2abup/osfstorage/Features/ data/train
rm -r 2abup

echo "Downloading TSR train-data from osf, this may take a while"
osf -p 6etg8 clone
echo "Finished downloading TSR train-data now cleaning"
cp -r 6etg8/osfstorage/Features/* data/train
rm -r 6etg8

echo "Finished downloading data"
