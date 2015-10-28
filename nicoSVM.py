#!/usr/bin/env python

"""
@author: Nicolas Angelard-Gontier
"""

import pickle

# Uncomment the 1st time to generate obj/train.pkl & obj/test.pkl
# No need after that, can directly load the data from the pkl files.
"""
from nicoLoadTestData import getTestData, getTrainData

test = getTestData(option=4)
print len(test)
print test["1994"]

train = getTrainData(option=4)
print len(train)
print train["2015"]


def save_obj(obj, name):
    with open('obj/'+name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

print "saving test..."
save_obj(test, "test")

print "saving train..."
save_obj(train, "train")
"""

def load_obj(name):
    with open('obj/'+name+'.pkl', 'rb') as f:
        return pickle.load(f)

print "fetching test..."
test = load_obj("test")
print len(test)
print test["1994"]

print "fetching train..."
train = load_obj("train")
print len(train)
print train["2015"]


#TODO: do the SVM now!
