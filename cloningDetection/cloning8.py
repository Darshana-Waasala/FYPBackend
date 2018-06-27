"""the intended steps of execution
1. get the slic segments
2. iterate over the unique elements of the segments
    2.1 get the closed area/patch for each segment
    2.2 extract SIFT key points of each segment separately and store them in a dictionary
    => dictionary = {segmentValue:Feature,...}
    => class Feature:
            keys-
            description-
3. double iterate over dictionary to get (nxn-n!) iterations
    3.1 match key points of each segemnt with that of other segments
    if match found under the threshold then mark as cloned"""

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


class Features:
    def __init__(self, keys: list, des: numpy.ndarray):
        """this is the constuctor of the method"""
        self.keys = keys
        self.des = des


class Channels:
    def __init__(self,red:int, green:int, blue:int):
        """this will help to hold the average value of three channel pixels in a segment"""
        self.red = red
        self.green = green
        self.blue = blue


def getImageSegments(rgbImage: numpy.uint8, segments: int, sigma: int) -> numpy.ndarray:
    """this will return the slic segmetns"""
    # img = img_as_float(image)
    bgrImage = rgbImage[:, :, ::-1]  # converted to bgr order because skimage work on bgr images
    segmentedData = slic(image=bgrImage, n_segments=segments, sigma=sigma)
    # drawSegments(image=bgrImage, segments=segmentedData)
    return segmentedData


def getMostAppropriteSegementNumber(image: numpy.ndarray) -> int:
    """shape is in the order ROWS,COLS,CHANNELS -> (y,x,c)"""
    y, x, c = image.shape
    totalSegments = int(round((x * y) / (50 * 50)))
    return totalSegments


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


def getSIFTKeyDes(image: numpy.ndarray) -> (list, numpy.ndarray):
    """here we do the keypoint besed calculations"""
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keys, des = sift.detectAndCompute(grayImage, None)
    # print('type of keys : ', type(keys))
    # print('type of descriptors : ', type(des))
    # print('length of keys - of type <class \'list\'>: ', len(keys))
    # print('shape of descriptors - of type <class \'numpy.ndarray\'>: ', des.shape)
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


def getLocalizedImageSegment(segmentValue: int, segments: numpy.ndarray, image: numpy.ndarray) -> numpy.ndarray:
    """this will get the approximated image segment for the given segment mask value"""

    '''assuming the sent suspect keypoints are not in the same segment'''
    '''numpy.where will give the list of rows and list of cols corresponding each pixel in the segment 
    with the specified value'''
    rows, cols = numpy.where(segments == segmentValue)

    '''image ROI is taken as image[y1:y2,x1:x2]'''
    imageSegment = image[min(rows):max(rows), min(cols):max(cols)]
    return imageSegment


def getArrangedDictionary(segments: numpy.ndarray, image: numpy.ndarray) -> dict:
    """this will prepare a dictionary of key ponts and their descriptors of each approximated patches"""

    '''prepare the dictionary that is to be returned and is arranged as,
    dict{segmentValue:FeatureObject,....}'''
    dictionary = {}

    '''get the unique values of the segmentation mask'''
    uniqueSegmentValues = numpy.unique(segments)

    '''iterate over the segment to get the localized sift keypoints'''
    for segmentValue in uniqueSegmentValues:
        localImageSegment = getLocalizedImageSegment(segmentValue=segmentValue, segments=segments, image=image)
        keys, des = getSIFTKeyDes(image=localImageSegment)

        '''adding only if at least one key point is found'''
        if des is not None:
            dictionary[segmentValue] = Features(keys=keys, des=des)

    return dictionary


def getMatchedPatches(arrangedDictionary: dict, matchPerCluster=5) -> dict:
    """this function will iterate over the localized descriptors and return the dictionary of matched items"""
    # leastDistance = 5000

    '''prepare the BF matcher for the knn match between the localized keypoints'''
    bf = cv2.BFMatcher()  # BFMatcher with default params

    matchedSegments = {}  # the dictionary to hold the matched segments
    '''iterate over the arranged patches to get (nxn - n!) iterations'''
    for index1, segmentValue1 in enumerate(arrangedDictionary):
        for index2, segmentValue2 in enumerate(arrangedDictionary):

            '''to make sure that we check the segment combinations that have not met before'''
            condition = (index2 > index1) & (arrangedDictionary[segmentValue1].des is not None) & \
                        (arrangedDictionary[segmentValue2].des is not None)
            if condition:
                print('condition....', condition)
                matches = bf.knnMatch(queryDescriptors=arrangedDictionary[segmentValue1].des,
                                      trainDescriptors=arrangedDictionary[segmentValue2].des, k=2)
                # print('length of matches : ',len(matches))
                count = 0  # to keep the count of the matches under the threshold
                '''iterating over the matches to filter out those under the required threshold'''
                for i, m in enumerate(matches):

                    '''since some of the matches do not get both n and m'''
                    if len(m) > 1:
                        if m[0].distance < 0.4 * m[1].distance:
                            count += 1
                            '''if m[1].distance != 0:
                                if (m[0].distance / m[1].distance) < leastDistance:
                                    leastDistance = (m[0].distance / m[1].distance)'''

                '''filling the segment into the dictionary if there are required number of matches in the patches'''
                if count >= matchPerCluster:
                    if segmentValue1 in matchedSegments:
                        matchedSegments[segmentValue1].append(segmentValue2)
                    elif segmentValue2 in matchedSegments:
                        matchedSegments[segmentValue2].append(segmentValue1)
                    else:
                        matchedSegments[segmentValue1] = [segmentValue2]
            else:
                continue
    # print('least distance rechorded .......', leastDistance)
    return matchedSegments


def drawMatchedClusters(image: numpy.ndarray, matchedClusters: dict, segments: numpy.ndarray) -> None:
    """this will draw the matched clusters"""
    for matchedClusterNumber in matchedClusters:
        rows1, cols1 = numpy.where(segments == matchedClusterNumber)
        for row1, col1 in zip(rows1, cols1):
            image[row1, col1, 0] = 0  # (255, 255, 255)
        for similarClusterNumber in matchedClusters[matchedClusterNumber]:
            rows2, cols2 = numpy.where(segments == similarClusterNumber)

            ''' Draw a diagonal blue line with thickness of 5 px parameters: pt1 is in (x,y) order '''
            cv2.line(img=image, pt1=(cols1[0], rows1[0]), pt2=(cols2[0], rows2[0]), color=(255, 0, 0), thickness=2)

            for row2, col2 in zip(rows2, cols2):
                image[row2, col2, 1] = 0  # (255, 255, 255)

    cv2.imshow('clone detected image', resizeImage(image))


def getEnlargeLocalImageSegment(matchedPatches: dict, segments: numpy.ndarray, image: numpy.ndarray):
    """this will enlarge the image patch based on averageColor"""


def getAverageImageColorValues(image: numpy.ndarray, segment: numpy.ndarray) -> dict:
    """this will generate average color value per segment and return the result as a dictionary with
    segment value as key and average pixel value as value"""
    segmentAverageColorValue = {}
    uniqueSegmentValues = numpy.unique(segment)

    '''iterating over segment values to get the average corresponding image color values'''
    for segmentValue in uniqueSegmentValues:
        tempRedValue = 0
        tempGreenValue = 0
        tempBlueValue = 0
        rows, cols = numpy.where(segment == segmentValue)

        '''iterating over the pixels of the segment to get the average of three channels'''
        for row, col in zip(rows, cols):
            tempRedValue = tempRedValue + image[row, col, 0]
            tempGreenValue = tempGreenValue + image[row, col, 1]
            tempBlueValue = tempBlueValue + image[row, col, 2]
        segmentAverageColorValue[segmentValue] = Channels(red=int(round(tempRedValue/len(rows))),green=int(round(tempGreenValue/len(rows))),
                                                          blue=int(round(tempBlueValue/len(rows))))

        '''iterating over the segment to color the image'''
        for row,col in zip(rows,cols):
            image[row,col,0] = segmentAverageColorValue[segmentValue].red
            image[row, col, 1] = segmentAverageColorValue[segmentValue].green
            image[row, col, 2] = segmentAverageColorValue[segmentValue].blue

    cv2.imshow('new colored image',resizeImage(image=image))


""" ################################## execution start position ############################################## """
start_time = time.time()
# img = cv2.imread('/home/waasala/workspace/gimp/colorFlower-cloned.jpeg')
# img = cv2.imread('/home/waasala/workspace/gimp/DSC01176tamp14.jpg')
# img = cv2.imread('/home/waasala/workspace/gimp/colorFlower_rotated.jpeg')
img = cv2.imread('/home/waasala/workspace/gimp/athal.jpeg')

segments = getImageSegments(rgbImage=img, segments=getMostAppropriteSegementNumber(image=img), sigma=5)
print('got segments :', len(numpy.unique(segments)))
# arrangedDict = getArrangedDictionary(segments=segments, image=img)
# print('arranged dictionary', len(arrangedDict))
# matchedPatches = getMatchedPatches(arrangedDictionary=arrangedDict, matchPerCluster=5)
# print('matched patches:', len(matchedPatches))
# drawMatchedClusters(image=img, matchedClusters=matchedPatches, segments=segments)
getAverageImageColorValues(image=img,segment=segments)

print('time of execution - ', time.time() - start_time)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''some of the new things we have learned to use here:
print('the segments : ', numpy.unique(segments))
print('the minimum value: ', segments.min())
print('the maximum value: ', segments.max())

k = numpy.array([
    [1,2,3],
    [4,5,6]
])

print('test min :',k.min(), ' | max :',k.max(),' | segments :', numpy.unique(k))
'''
