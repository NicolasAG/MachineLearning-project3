#!/usr/bin/env python
import idx2numpy
import pprint

TRAIN_IMAGES = 'data_and_scripts/MNIST/train-images.idx3-ubyte'
TRAIN_LABELS = 'data_and_scripts/MNIST/train-labels.idx1-ubyte'
TEST_IMAGES = 'data_and_scripts/MNIST/t10k-images.idx3-ubyte'
TEST_LABELS = 'data_and_scripts/MNIST/t10k-labels.idx1-ubyte'

#cf: https://pypi.python.org/pypi/idx2numpy
#cf: http://yann.lecun.com/exdb/mnist/

# Reading
print "train images:"
train_images = idx2numpy.convert_from_file(TRAIN_IMAGES)
#for i in range(28):
#    print train_images[0][i]
print train_images
print train_images.shape

print "train labels:"
train_labels = idx2numpy.convert_from_file(TRAIN_LABELS)
print train_labels
print train_labels.shape

print "test images:"
test_images = idx2numpy.convert_from_file(TEST_IMAGES)
print test_images
print test_images.shape

print "test labels:"
test_labels = idx2numpy.convert_from_file(TEST_LABELS)
print test_labels
print test_labels.shape
"""
