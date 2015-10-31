#!/usr/bin/env python

"""
@author: Nicolas Angelard-Gontier
"""

import pickle
from sklearn import svm
from sklearn.decomposition import PCA
from datetime import datetime
import numpy as np

def save_obj(obj, name):
    with open('obj/'+name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/'+name+'.pkl', 'rb') as f:
        return pickle.load(f)


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

print "saving test..."
save_obj(test, "test")

print "saving train..."
save_obj(train, "train")
"""

print "fetching train..."
train = load_obj("train")

#print "fetching test..."
#test = load_obj("test")


###########################
### SVM on regular data ###
###########################
TRAIN_SIZE = 8000
TEST_SIZE = 2000
DO_PCA = 784
BINARY_IMAGE = False
"""
modify data:
    change values to be either 0 or 1.

    skimage.filters.gabor(image, frequency,...)
"""

"""
This methods applies thresholding to a given image array.
@param image_array - the array of pixel values corresponding to the image.
@param threshold - the threshold value that decides if the pixel is going to be black or white.
@return - the modified image.
"""
def binaryImage(image_array, threshold=0.5):
    for i in range(len(image_array)):
        if image_array[i] <= 0.5:
            image_array[i] = 0.0
        else:
            image_array[i] = 1.0
    return image_array

###
# Generating X and Y matrices.
###
X = []
Y = []
print "generating X & Y..."
for image in train.values()[:TRAIN_SIZE+TEST_SIZE]: #max: 50000
    if BINARY_IMAGE:
        X.append(binaryImage(image["0"]))    # train on original image
        X.append(binaryImage(image["90"]))   # train on 90* rotated image
        X.append(binaryImage(image["180"]))  # train on 180* rotated image
        X.append(binaryImage(image["270"]))  # train on 270* rotated image
    else:
        X.append(image["0"])    # train on original image
        X.append(image["90"])   # train on 90* rotated image
        X.append(image["180"])  # train on 180* rotated image
        X.append(image["270"])  # train on 270* rotated image
    Y.extend([image["##"]]*4) # add the target value 4 times.
assert len(X) == 4*(TRAIN_SIZE+TEST_SIZE)
assert len(Y) == 4*(TRAIN_SIZE+TEST_SIZE)

if DO_PCA > 0 and DO_PCA <= 2304:
    pca = PCA(n_components=DO_PCA) # max=2304=48*48  784=28*28
    X = pca.fit_transform(X)

X = np.asarray(X)
Y = np.asarray(Y)
if not X.flags['C_CONTIGUOUS']:
    print "WARNING: X is not C-ordered contiguous."
if not Y.flags['C_CONTIGUOUS']:
    print "WARNING: Y is not C-ordered contiguous."

CGAMMA = [0.1,0.5,1,5,10]
COEF = [0,1,5,10]
DEG = [2,3,4,5]

for c in CGAMMA:
    for gamma in CGAMMA:

        start = datetime.now()
        print "RBF ; C=%f ; gamma=%f" % (c, gamma)

        ###
        # Training classifier.
        ###
        print "training the classifier..."
        classifier = svm.SVC(
            cache_size=1000,
            kernel='rbf',   # try 'rbf', 'poly', 'sigmoid'
            C=c,            # try 0.1, 0.5, 1, 5, 10
            gamma=gamma,    # try 0.1, 0.5, 1, 5, 10
            coef0=0.0,      # try 0.0, 1.0, 5, 10
            degree=3.0,     # try 2.0, 3.0, 4, 5
        ).fit(X[:TRAIN_SIZE*4],Y[:TRAIN_SIZE*4]) # train on the first TRAIN_SIZE points.

        ###
        # Making predictions & write to file.
        ###
        print "making predictions..."
        correct = 0.0

        """Uncomment to write to file"""
        #test_output_file = open('data_and_scripts/test_output.csv', "wb")
        #writer = csv.writer(test_output_file, delimiter=',') 
        #writer.writerow(['Id', 'Prediction']) # write header
        #for idx in test.keys():
        for i in range(0, TEST_SIZE*4, 4): # i goes form 0 to TEST_SIZE*4 by steps of 4
            prediction1 = classifier.predict(X[(TRAIN_SIZE*4)+i])[0]   # predict on the original image
            prediction2 = classifier.predict(X[(TRAIN_SIZE*4)+i+1])[0] # predict on the 90* rotated image
            prediction3 = classifier.predict(X[(TRAIN_SIZE*4)+i+2])[0] # predict on the 180* rotated image
            prediction4 = classifier.predict(X[(TRAIN_SIZE*4)+i+3])[0] # predict on the 270* rotated image
            counts = np.bincount([prediction1,prediction2,prediction3,prediction4])
            prediction = np.argmax(counts)
            
            if prediction == Y[(TRAIN_SIZE*4)+i]: # take the most popular prediction.
                correct += 1.0
            #row = [idx, prediction]
            #writer.writerow(row)
        #test_output_file.close()
        print correct / TEST_SIZE

        print datetime.now() - start
        print "-----------------------------------------"


