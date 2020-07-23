# Computational target inference by mining transcriptional data using a novel Siamese spectral-based graph convolutional network
SSGCN

# System requirements
## Operating systems  requirements
This package is supported for  Linux. The package has been tested on Linux: Ubuntu 16.04
## Software Dependencies
The SSGCN model was implemented in the TensorFlow framework (version TensorFlow-GPU 1.14.0) in Python 3.7.3.Lower version of python may cause the program to not work
## Hardware requirements
The SSGCN requires a computer with a  GPU.
# Installation guide:
## Instructions

conda install tensorflow-gpu=1.14

git clone https://github.com/boyuezhong/SSGCN.git.
## Typical install time 
Installation takes 3 minutes.

# Demo
cd SSGCN

python SSGCN.py
## Expected output
The  expected output can be found in  ./test_10_12
## Expected run time 
 Inferring the target of a compound took 20 second on a NVIDIA TITAN RTX GPU .

# Instructions for use How to run the software
python SSGCN.py


# Reproduction instructions
The results of benchmark can be reproduced.
# License
This code  is  licensed  under the Apache 2.0 License.

