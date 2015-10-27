#!/usr/bin/env python

"""
@author: Nicolas Angelard-Gontier
"""

from nicoLoadTestData import getTestData

test = getTestData()
print len(test)
print test["1994"]
print len(test["1994"])

#TODO: do the SVM now!
