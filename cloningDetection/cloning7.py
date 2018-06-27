"""this is working sample of cloning6.py -> same thing
modified to automatically test for a given data set"""

# import the necessary packages
import threading
from ctypes import c_void_p

from cv2.cv2 import HOGDescriptor
from skimage import img_as_float
from skimage.segmentation import slic
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import numpy
import cv2
import scipy.spatial.distance
import time
import glob


class Data:
    def __init__(self, x1=None, y1=None, x2=None, y2=None):
        """this act as the constructor"""
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class KeyDes:
    def __init__(self, keypoint=None, descriptor=None):
        """this act as the constructor"""
        self.keypoint = keypoint
        self.descriptor = descriptor


class SegVal:
    """this class is used to store HOG descriptors in a dictionary with duplicate key values"""

    def __init__(self, segmentValue):
        self.segmentValue = segmentValue


def getImageSegments(rgbImage: numpy.uint8, segments: int, sigma: int) -> numpy.ndarray:
    """this will return the slic segmetns"""
    # img = img_as_float(image)
    bgrImage = rgbImage[:, :, ::-1]  # converted to bgr order because skimage work on bgr images
    segmentedData = slic(image=bgrImage, n_segments=segments, sigma=sigma)
    # drawSegments(image=bgrImage, segments=segmentedData)
    return segmentedData


def getSIFTKeyDes(image: numpy.uint8) -> (list, numpy.ndarray):
    """here we do the keypoint besed calculations"""
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keys, des = sift.detectAndCompute(grayImage, None)
    # print('type of keys : ', type(keys))
    # print('type of descriptors : ', type(des))
    print('length of keys - of type <class \'list\'>: ', len(keys))
    print('shape of descriptors - of type <class \'numpy.ndarray\'>: ', des.shape)
    return keys, des


def clusterKeypoints(keypoints: list, descriptors: numpy.ndarray, segments: numpy.ndarray, keysPerCluster=1) -> dict:
    """will add the keys to a dictionary only of the segment have key points then the required threshold"""
    dictionary = {}
    for keypoint, descriptor in zip(keypoints, descriptors):  # index is starting from ZERO
        segmentVal = segments[int(round(keypoint.pt[1]))][int(round(keypoint.pt[0]))]
        '''segments are in (row,col) order and keypoint.pt is in (x,y) order'''

        if segmentVal in dictionary:
            dictionary[segmentVal].append(KeyDes(keypoint=keypoint, descriptor=descriptor))
        else:
            dictionary[segmentVal] = [KeyDes(keypoint=keypoint, descriptor=descriptor)]

    '''removing the segments those that have lesser key points than required'''
    keysInDictionary = [*dictionary]
    for key in keysInDictionary:
        if len(dictionary[key]) < keysPerCluster:
            del dictionary[key]

    return dictionary


def matchKeypointsBFSIFT(description: numpy.ndarray, segments: numpy.ndarray, threshold=15) -> dict:
    # BFMatcher with default params
    bf = cv2.BFMatcher()

    '''separate the keypoints into two groups to use them in the knn clustering'''
    matches = bf.knnMatch(description, description, k=2)

    # ratio test as per Lowe's paper
    count = 0
    bestMatches = {}  # dictionary to hold the cluster matches
    for i, (m, n) in enumerate(matches):
        '''n has the higher value than m (always - did not encounter other way round)'''
        if abs(n.distance - m.distance) < threshold:
            count += 1

            x1 = int(round(keys[m.queryIdx].pt[0]))
            y1 = int(round(keys[m.queryIdx].pt[1]))
            x2 = int(round(keys[n.trainIdx].pt[0]))
            y2 = int(round(keys[n.trainIdx].pt[1]))

            segmentVal1 = segments[y1][x1]
            segmentVal2 = segments[y2][x2]
            '''segments are in (row,col) order and keypoint.pt is in (x,y) order and are in floating points'''

            '''filling the matched segments into the best matches dictionary'''
            if segmentVal1 in bestMatches:
                bestMatches[segmentVal1].append(segmentVal2)
            elif segmentVal2 in bestMatches:
                bestMatches[segmentVal2].append(segmentVal1)
            else:
                bestMatches[segmentVal1] = [segmentVal2]

    return bestMatches


def getMostAppropriteSegementNumber(image: numpy.ndarray) -> int:
    """shape is in the order ROWS,COLS,CHANNELS -> (y,x,c)"""
    y, x, c = image.shape
    totalSegments = int(round((x * y) / (50 * 50)))
    return totalSegments


""" ################################## execution start position ############################################## """
job_start = time.time()
allFileList = glob.glob('/home/waasala/workspace/gimp/*')

thresholdForSIFT = 15
requiredKeypointsPerCluster = 1
thresholdForHOG = 15

cloned_log = open('/home/waasala/workspace/cloned.txt', 'a')
no_cloned_log = open('/home/waasala/workspace/no_clone.txt', 'a')

for i, file in enumerate(allFileList):

    img = cv2.imread(file)
    start_time = time.time()

    keys, des = getSIFTKeyDes(image=img)
    segments = getImageSegments(rgbImage=img.copy(), segments=getMostAppropriteSegementNumber(image=img), sigma=5)

    bestMatches = matchKeypointsBFSIFT(des, segments, 15)

    writing_string = str(i) + ') -> file name: ' + str(file) + ' | time taken:' + str(
        time.time() - start_time) + ' | keys:' + str(
        len(keys)) + ' | segments:' + str(len(segments)) + '\n'
    if len(bestMatches) > 0:
        cloned_log.write(writing_string)
    else:
        no_cloned_log.write(writing_string)

final_string = '\n\n>>>>>>>>>>>>>>>>>>>>>>total time taken:' + str(time.time() - job_start) + '>>>>>>>>>>>>>>>>>>>>>>'
cloned_log.write(final_string)
no_cloned_log.write(final_string)

cloned_log.close()
no_cloned_log.close()
""""'''steps for the HOG detection'''
segsWithoutSIFTs = getSegmentsWithoutRequiredNumberOfKeys(keys=keys, segments=segments,
                                                          requiredKeysPreCluster=requiredKeypointsPerCluster)
matchedSegsWithHOG = matchWithHOGKNN(noSIFTkeySegValList=segsWithoutSIFTs, segments=segments, imageColor=img,
                                             threshold=thresholdForHOG)
print('detected pathces form HOG : ', len(matchedSegsWithHOG))

bestMatches.update(matchedSegsWithHOG)
print('total patches',len(bestMatches))
cv2.imshow('final image', resizeImage(img))
drawMatchedClusters(image=img, matchedClusters=bestMatches, segments=segments)"""

"""general knowledge
in python opencv image point can be accessed as -> img[10,100] = (255,255,255) # x=100,y=10
                                this is similar to -> img[x,y] = (255,255,255)"""
