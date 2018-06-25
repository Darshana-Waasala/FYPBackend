"""this is working for cloning tool but not for geometric oriented cloning
cloning is detected using - SIFT keys, FLANN|BF for matching"""

# import the necessary packages
import threading

from cv2.cv2 import HOGDescriptor
from skimage import img_as_float
from skimage.segmentation import slic
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import numpy
import cv2
import scipy.spatial.distance
import time


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


def getImageSegments(rgbImage: numpy.uint8, segments: int, sigma: int) -> numpy.ndarray:
    """this will return the slic segmetns"""
    # img = img_as_float(image)
    bgrImage = rgbImage[:, :, ::-1]  # converted to bgr order because skimage work on bgr images
    segmentedData = slic(image=bgrImage, n_segments=segments, sigma=sigma)
    # drawSegments(image=bgrImage, segments=segmentedData)
    return segmentedData


def drawSegments(image: numpy.float, segments: numpy.ndarray) -> None:
    # show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments" % (50))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments))
    plt.axis("off")
    # show the plots
    plt.show()
    return None


def resizeImage(image: numpy.uint8) -> numpy.uint8:
    """this will resize image if dimensions are greter than 512"""
    if image.shape[0] > 512:
        width = int(numpy.around((image.shape[1]) / 2.5))
        height = int(numpy.around(image.shape[0] / 2.5))
        resizedImage = cv2.resize(src=image, dsize=(width, height))
        return resizedImage
    return image


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


def drawSIFTKeysOnly(image: numpy.uint8) -> None:
    """this will calculate only the SIFT keypoints only"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)
    # img = cv2.drawKeypoints(gray, kp, image)
    img = cv2.drawKeypoints(image=image, keypoints=kp, outImage=image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                            color=(51, 163, 236))
    cv2.imshow('sift_keypoints.jpg', img)
    return None


def drawSURFKeysOnly(image: numpy.uint8) -> None:
    """this will calculate only the SURF keypoints only"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create(1000)
    kp = surf.detect(gray, None)
    img = cv2.drawKeypoints(image=image, keypoints=kp, outImage=image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                            color=(51, 163, 236))
    cv2.imshow('sift_keypoints.jpg', img)
    return None


def clusterKeypoints(keypoints: list, descriptors: numpy.ndarray, segments: numpy.ndarray) -> dict:
    dictionary = {}
    for keypoint, descriptor in zip(keypoints, descriptors):  # index is starting from ZERO
        segmentVal = segments[int(round(keypoint.pt[1]))][int(round(keypoint.pt[0]))]
        '''segments are in (row,col) order and keypoint.pt is in (x,y) order'''

        if segmentVal in dictionary:
            dictionary[segmentVal].append(KeyDes(keypoint=keypoint, descriptor=descriptor))
        else:
            dictionary[segmentVal] = [KeyDes(keypoint=keypoint, descriptor=descriptor)]
    return dictionary


def getDescriptorListForSegment(segmentValue: int, dictionary: dict) -> numpy.ndarray:
    descriptors = []
    for item in dictionary[segmentValue]:
        descriptors.append(item.descriptor)
    return numpy.float32(descriptors)


def identifySuspectAreas(minThreshKeyPair: Data, segments: numpy.ndarray, image: numpy.ndarray) -> dict:
    """this will return array of numpy.ndarray of matched areas"""
    '''assuming the sent suspect keypoints are not in the same segment'''
    rows1, cols1 = numpy.where(segments == minThreshKeyPair.y1)
    rows2, cols2 = numpy.where(segments == minThreshKeyPair.y2)

    area1 = image[min(rows1):max(rows1), min(cols1):max(cols1)]  # [y1:y2, x1:x2]
    area2 = image[min(rows2):max(rows2), min(cols2):max(cols2)]  # [y1:y2, x1:x2]

    suspectArea = {'area1': area1, 'area2': area2}
    return suspectArea


def calculsteHOGofSuspectAreas(suspectAreas: dict):
    hog = HOGDescriptor()
    returnVAl = hog.compute(img=suspectAreas.area1)
    print(type(returnVAl))


def drawMatchedClusters(image: numpy.uint8, matchedClusters: dict, segments: numpy.ndarray) -> None:
    """this will draw the matched clusters"""
    for matchedClusterNumber in matchedClusters:
        rows, cols = numpy.where(segments == matchedClusterNumber)
        for row, col in zip(rows, cols):
            image[row, col] = (255, 255, 255)
        for similarClusterNumber in matchedClusters[matchedClusterNumber]:
            rows, cols = numpy.where(segments == similarClusterNumber)
            for row, col in zip(rows, cols):
                image[row, col] = (255, 255, 255)

    cv2.imshow('clone detected image', resizeImage(image))


def matchKeypointsFlannSIFT(description: numpy.ndarray, segments: numpy.ndarray) -> list:
    """this will match the matches as a list
    :type description: SIFT descriptors of the image
    :rtype: list of best matches
    """
    # FLANN parameters                                                                                                     
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    '''separate the keypoints into two groups to use them in the knn clustering'''
    # splitedDescriptions = numpy.array_split(description, 2)
    # desRange1 = len(splitedDescriptions[0])

    # matches = flann.knnMatch(splitedDescriptions[0], splitedDescriptions[1], k=2)
    matches = flann.knnMatch(description, description, k=2)
    '''the parameters -> query feature, train feature, cluster size'''
    print('length of matches : ', len(matches), ' type of matches -----  ', type(matches[0][0]))

    # ratio test as per Lowe's paper                                                                                       
    count = 0
    bestMatches = {}  # dictionary to hold the cluster matches

    for i, (m, n) in enumerate(matches):
        '''n has the higher value than m (always - did not encounter other way round)'''
        # if (n.distance != 0):
        # if m.distance < 0.6 * n.distance:
        if n.distance - m.distance < 15:
            count += 1
            # x1 = int(round(keys[m.queryIdx].pt[0]))
            # y1 = int(round(keys[m.queryIdx].pt[1]))
            # x2 = int(round(keys[m.trainIdx + desRange1].pt[0]))
            # y2 = int(round(keys[m.trainIdx + desRange1].pt[1]))
            x1 = int(round(keys[m.queryIdx].pt[0]))
            y1 = int(round(keys[m.queryIdx].pt[1]))
            x2 = int(round(keys[m.trainIdx].pt[0]))
            y2 = int(round(keys[m.trainIdx].pt[1]))

            # print('(', x1, ',', y1, ')', '-(', x2, ',', y2, ')')
            # print(' m query index = ', m.queryIdx, '| m train index = ', m.trainIdx,
            #       '| n query index = ', n.queryIdx, '| n train index = ', n.trainIdx,
            #       '| n distance - ', n.distance, '| m distance - ', m.distance, )

            '''draw patches around the keypoints matched'''
            img[y1 - 9:y1 + 9, x1 - 9:x1 + 9, 1] = 0
            img[y2 - 9:y2 + 9, x2 - 9:x2 + 9, 2] = 0

            ''' Draw a diagonal blue line with thickness of 5 px parameters: pt1 is in (x,y) order '''
            cv2.line(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=1)

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

    print('total count : ', count)

    return bestMatches


def matchKeypointsBFSIFT(description: numpy.ndarray, segments: numpy.ndarray) -> list:
    # BFMatcher with default params
    bf = cv2.BFMatcher()

    '''separate the keypoints into two groups to use them in the knn clustering'''
    # splitedDescriptions = numpy.array_split(description, 2)
    # desRange1 = len(splitedDescriptions[0])
    # print('testing.....',description[desRange1] == splitedDescriptions[1][0])

    # matches = bf.knnMatch(splitedDescriptions[0], splitedDescriptions[1], k=2)
    matches = bf.knnMatch(description, description, k=2)

    # ratio test as per Lowe's paper
    count = 0
    bestMatches = {}  # dictionary to hold the cluster matches
    minimumDistance=100000
    for i, (m, n) in enumerate(matches):
        '''n has the higher value than m (always - did not encounter other way round)'''
        # if (n.distance != 0):
        # if m.distance < 0.75 * n.distance:
        if abs(n.distance-m.distance)<minimumDistance:
            minimumDistance =abs(n.distance-m.distance)
        if abs(n.distance - m.distance) < 15:
            count += 1
            # print('distance : ', (m.distance / n.distance), m.trainIdx, n.trainIdx)

            # x1 = int(round(keys[m.queryIdx].pt[0]))
            # y1 = int(round(keys[m.queryIdx].pt[1]))
            # x2 = int(round(keys[m.trainIdx + desRange1].pt[0]))
            # y2 = int(round(keys[m.trainIdx + desRange1].pt[1]))
            x1 = int(round(keys[m.queryIdx].pt[0]))
            y1 = int(round(keys[m.queryIdx].pt[1]))
            x2 = int(round(keys[n.trainIdx].pt[0]))
            y2 = int(round(keys[n.trainIdx].pt[1]))
            # print('(', x1, ',', y1, ')', '-(', x2, ',', y2, ')')
            print(' m query index = ', m.queryIdx, '| m train index = ', m.trainIdx,
                  '| n query index = ', n.queryIdx, '| n train index = ', n.trainIdx,
                  '| n distance - ', n.distance, '| m distance - ', m.distance, )

            '''draw patches around the keypoints matched'''
            img[y1 - 9:y1 + 9, x1 - 9:x1 + 9, 1] = 0
            img[y2 - 9:y2 + 9, x2 - 9:x2 + 9, 2] = 0

            ''' Draw a diagonal blue line with thickness of 5 px parameters: pt1 is in (x,y) order '''
            cv2.line(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=1)

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

    print('total count :', count, '| minimum distance :', minimumDistance)

    return bestMatches


def matchKeypointsBFORB(image:numpy.ndarray,segments:numpy.ndarray)->list:

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    kp, des = orb.detectAndCompute(image, None)
    print('total number of ORB key points : ', len(kp))
    splitedDescriptions = numpy.array_split(des, 2)
    desRange1 = len(splitedDescriptions[0])

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    # matches = bf.match(des, des)
    matches = bf.match(splitedDescriptions[0], splitedDescriptions[1])
    print('total number of matches : ',len(matches))

    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    for match in matches:
        print('match:',match, '| match distance:',match.distance, ' | match trainIdx:',
              match.trainIdx, ' | match queryIdx', match.queryIdx)



def getMostAppropriteSegementNumber(image:numpy.ndarray)->int:
    """shape is in the order ROWS,COLS,CHANNELS -> (y,x,c)"""
    y,x,c = image.shape
    totalSegments = int(round((x*y)/(50*50)))
    print('total segements : ',totalSegments)
    return totalSegments

""" ################################## execution start position ############################################## """
start_time = time.time()
# img = cv2.imread('/home/waasala/workspace/PycharmProjects/OpenCVBasic/cloningDetection/image.png')
# img = cv2.imread('/home/waasala/workspace/gimp/colorFlower-cloned.jpeg')
# img = cv2.imread('/home/waasala/workspace/gimp/00007tamp4.jpg')
# img = (cv2.imread('/home/waasala/workspace/gimp/P1000472tamp5.jpg'))
# img = resizeImage(cv2.imread('/home/waasala/workspace/gimp/P1000293tamp9.jpg'))
# img = (cv2.imread('/home/waasala/workspace/gimp/DSC_0095_01_cloned.jpg'))
# img = cv2.imread('/home/waasala/workspace/gimp/gardenMultipleClone.jpg')
img = cv2.imread('/home/waasala/workspace/gimp/colorFlower_rotated.jpeg')

keys, des = getSIFTKeyDes(image=img)
segments = getImageSegments(rgbImage=img.copy(), segments=getMostAppropriteSegementNumber(image=img), sigma=5)
# matchKeypointsBFORB(image=img,segments=segments)
# bestMatches = matchKeypointsFlannSIFT(des, segments)
bestMatches = matchKeypointsBFSIFT(des, segments)
cv2.imshow('final image', resizeImage(img))
# drawMatchedClusters(image=img, matchedClusters=bestMatches, segments=segments)

print('time of execution - ', time.time() - start_time)

cv2.waitKey(0)
cv2.destroyAllWindows()

"""general knowledge
in python opencv image point can be accessed as -> img[10,100] = (255,255,255) # x=100,y=10
                                this is similar to -> img[x,y] = (255,255,255)"""
