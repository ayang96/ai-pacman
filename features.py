# features.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import numpy as np
import util
import samples

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def basicFeatureExtractor(datum):
    """
    Returns a binarized and flattened version of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features indicating whether each pixel
            in the provided datum is white (0) or gray/black (1).
    """
    features = np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1
    return features.flatten()

def enhancedFeatureExtractor(datum):
    """
    Returns a feature vector of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features designed by you. The features
            can have any length.

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...
       The extra feature added indicates if this image contains a dark number loop.
       For example, digits 4,6,8,9 have a number loop. 
       Namely a white area surrounded by dark pixels. 
       The algorithm uses a depth-frist search method to check if a white spot is surround by dark pixels.
    ##
    """
    features = basicFeatureExtractor(datum)  #The existing method use each pixel position as a feature.

    "*** YOUR CODE HERE ***"
#    util.raiseNotDefined()
#
#    return features
    #Q9: Question 9
    #(rowDim, colDim)=datum.shape
    rowDim=DIGIT_DATUM_WIDTH
    colDim=DIGIT_DATUM_HEIGHT
    #Define a depth first search algorithm to check if a dark loop surrounds pixel (x,y)
    def dfs(x,y):
        if x<0 or x>=rowDim: #hit boundary, then it is not covered by a loop
            return False
        if y<0 or y>=colDim: #hit boundary, then it is not covered by a loop
            return False
    	if datum[x,y]>0: #itself is a dark pixel, that is fine.
            return True
        if (x,y) in visited.keys():# meet the previously visited pixel area, that is fine.
            return True
	#This is a white pixel, mark it visited. Then return true only if it surrounded by a dark loop
        visited[(x,y)]=True   
        return dfs(x+1,y) and dfs(x-1,y) and dfs(x,y+1) and dfs(x,y-1)

    #Check if this image contains a connected loop
    numberLoop=False
    x=0
    while x< rowDim and not numberLoop:
	y=0
        while y < colDim and not numberLoop:
            visited={}
	    #if it is a white pixel, surrounded by a dark loop
            if datum[x,y]==0 and dfs(x,y): 
                    numberLoop=True
	    y=y+1
	x=x+1
    #Now we need to expand the feature list, and add one extra feature.
    #Since it is a numpy array, we need to use concatenate call to expand

    extra=np.array([numberLoop])
    newfeatures=np.concatenate((features, extra))
    return newfeatures



def analysis(model, trainData, trainLabels, trainPredictions, valData, valLabels, validationPredictions):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the print_digit(numpy array representing a training example) function
    to the digit

    An example of use has been given to you.

    - model is the trained model
    - trainData is a numpy array where each row is a training example
    - trainLabel is a list of training labels
    - trainPredictions is a list of training predictions
    - valData is a numpy array where each row is a validation example
    - valLabels is the list of validation labels
    - valPredictions is a list of validation predictions

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    # for i in range(len(trainPredictions)):
    #     prediction = trainPredictions[i]
    #     truth = trainLabels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         print_digit(trainData[i,:])


## =====================
## You don't have to modify any code below.
## =====================

def print_features(features):
    str = ''
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    for i in range(width):
        for j in range(height):
            feature = i*height + j
            if feature in features:
                str += '#'
            else:
                str += ' '
        str += '\n'
    print(str)

def print_digit(pixels):
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    pixels = pixels[:width*height]
    image = pixels.reshape((width, height))
    datum = samples.Datum(samples.convertToTrinary(image),width,height)
    print(datum)

def _test():
    import datasets
    train_data = datasets.tinyMnistDataset()[0]
    for i, datum in enumerate(train_data):
        print_digit(datum)

if __name__ == "__main__":
    _test()
