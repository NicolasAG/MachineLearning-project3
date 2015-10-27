"""
@author: Nicolas Angelard-Gontier
"""

import numpy as np
import csv


"""
Read the test csv file and returns a dictionary containing the test images as arrays
@return test_inputs is of the form: {
	"image ID 1" : {
		"0"  		: [...], <-- array of pixel values for unmodified test image.
		"90" 		: [...], <-- array of pixel values for the rotation by 90* to the right of the test image.
		"180"		: [...], <-- array of pixel values for the rotation by 180* of the test image.
		"270"		: [...], <-- array of pixel values for the rotation by 270* to the right of the test image.
	},
	"image ID 2" : {
		...
	},
	...
}
"""
def getTestData():
	KAGGLE_TEST = "data_and_scripts/test_inputs.csv"
	print "loading test data..."
	# Load all test inputs to a python list
	test_inputs = {}
	with open(KAGGLE_TEST, 'rb') as csvfile:
	    reader = csv.reader(csvfile, delimiter=',')
	    next(reader, None)  # skip the header
	    for test_input in reader:
	    	test_inputs[test_input[0]] = {} # for each image, we have a dictionary.
	        test_input_no_id = []
	        for pixel in test_input[1:]: # Start at index 1 to skip the Id
	            test_input_no_id.append(float(pixel))
	        test_inputs[test_input[0]]['0'] = np.asarray(test_input_no_id) # Store the array in the map
	        test_inputs[test_input[0]]['90'] = get90(test_input_no_id, 48, 48)
	        test_inputs[test_input[0]]['180'] = get180(test_input_no_id)
	        test_inputs[test_input[0]]['270'] = get270(test_input_no_id, 48, 48)

	print "done loading test data."
	print len(test_inputs)
	return test_inputs

"""
Returns the image array rotated by 180*.
@param image_array - the array pixel values for the image.
@return the reversed of the given array.
"""
def get180 (image_array):
	return np.asarray(image_array[::-1])

"""
Returns the image array rotated by 90* to the right.
@param image_array - the array pixel values for the image.
@param image_width - the number of 'columns' of pixels in the image.
@param image_height - the number of 'lines' of pixels in the image.
@return the array corresponding to the 90* rotated image.
"""
def get90 (image_array, image_width, image_height):
	a = []
	image = np.asarray(image_array).reshape(image_height, image_width)
	for j in range(image_width):
		for i in range(image_height)[::-1]:
			a.append(image[i][j])
	return np.asarray(a)


"""
Returns the image array rotated by 270* to the right.
@param image_array - the array pixel values for the image.
@param image_width - the number of 'columns' of pixels in the image.
@param image_height - the number of 'lines' of pixels in the image.
@return the array corresponding to the 270* rotated image (90+180).
"""
def get270 (image_array, image_width, image_height):
	return np.asarray(get180(get90(image_array, image_width, image_height)))


"""
uncomment to test
"""
#image = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
#print get180(image)
#print get90(image, 4, 4)
#print get270(image, 4, 4)

