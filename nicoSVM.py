#!/usr/bin/env python

"""
@author: Nicolas Angelard-Gontier
"""

import pickle
from sklearn import svm

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

#print "fetching test..."
#test = load_obj("test")
#print len(test)
#print test["1994"]

#print "fetching train..."
#train = load_obj("train")
#print len(train)
#print train["2015"]


###################
### SVM example ###
###################
#cf: http://scikit-learn.org/stable/modules/svm.html
X = [
    [0.0,0.9,0.0,
    0.5,0.7,0.0,   #<-- custom 1
    0.0,0.8,0.0],

    [0.0,0.9,0.0,
    0.7,0.1,0.8,   #<-- custom 0
    0.0,0.8,0.0],

    [0.0,0.9,0.0,
    0.5,0.1,0.8,   #<-- custom 2
    0.0,0.8,0.7]
]
Y = [1,0,2]

classifier = svm.SVC().fit(X,Y)
print classifier
print classifier.predict(
    [0.0,0.8,0.0,
    0.8,0.0,0.8,   #<-- custom 0
    0.0,0.8,0.0],
)
print classifier.predict(
    [0.0,0.7,0.0,
    0.7,0.1,0.5,   #<-- custom 2
    0.0,0.8,0.5]
)
print classifier.predict(
    [0.0,0.8,0.0,
    0.8,0.5,0.1,   #<-- custom 1
    0.0,0.7,0.0]    
)


"""
==================
PARAMETERS TO SET:
==================
classifier = Svm.SVC(
    C=1.0,
    cache_size=200,
    kernel='rbf',
    coef0=0.0, degree=3, gamma=0.0,
    probability=False
).fit(X,Y)

----------------------
C: default = 1
If you have a lot of noisy observations you should decrease it.
It corresponds to regularize more the estimation.
----------------------
cache_size: default = 200
The size of the kernel cache has a strong impact on run times for larger problems.
If you have enough RAM available, it is recommended to set cache_size to a higher value than the default of 200(MB),
 such as 500(MB) or 1000(MB).
----------------------
class_weight: default = None
If data for classification are unbalanced (e.g. many positive and few negative),
 set class_weight='auto' and/or try different penalty parameters C.
----------------------
kernel: default = 'rbf'
The kernel function can be any of the following:
linear,
polynomial: (gamma(x,x')+r)^d. d is specified by 'degree', r by 'coef0',
rbf: exp(-gamma|x-x'|^2). gamma is specified by 'gamma' (>0) ,
sigmoid: (tanh(gamma(x,x')+r)), where r is specified by 'coef0'.
----------------------
coef0: default = 0.0
(see kernel)
----------------------
degree: default = 3
(see kernel)
----------------------
gamma: default = 0.0
(see kernel)
----------------------
probability: default = False
When set to True, class membership probability estimates (from the methods predict_proba and predict_log_proba) are enabled.

"""

###########################
### SVM on regular data ###
###########################
X = []
Y = []

classifier = svm.SVC().fit(X,Y)
print classifier.predict()

