"""this is not completed. but intended to have the euclidean based feature matching and do HOG descriptor matching
if no match was found on SIFT features"""
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
        width = int(numpy.around((image.shape[1]) / 2))
        height = int(numpy.around(image.shape[0] / 2))
        resizedImage = cv2.resize(src=image, dsize=(width, height))
        return resizedImage
    return image


def getSIFTKeyDes(image: numpy.uint8) -> (list, numpy.ndarray):
    """here we do the keypoint besed calculations"""
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keys, des = sift.detectAndCompute(grayImage, None)
    print('type of keys : ', type(keys))
    print('type of descriptors : ', type(des))
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
        # segments are in (row,col) order and keypoint.pt is in (x,y) order

        if segmentVal in dictionary:
            dictionary[segmentVal].append(KeyDes(keypoint=keypoint, descriptor=descriptor))
        else:
            dictionary[segmentVal] = [KeyDes(keypoint=keypoint, descriptor=descriptor)]
    return dictionary


def matchedKeypoints(keypoints: list, descriptors: numpy.ndarray, threshold=40) -> list:
    """this function will return the matched within a threshold"""
    matchedKeypoints = []
    minDisVal = 100000;
    for index, keypoint in enumerate(keypoints):  # index is starting from ZERO
        for item in range(index + 1, (len(keypoints))):  # item is starting from 1
            doubleDisVal = scipy.spatial.distance.euclidean(descriptors[index], descriptors[item])
            if doubleDisVal < minDisVal:
                minDisVal = doubleDisVal
            if doubleDisVal < threshold:
                print(index, ' - ', doubleDisVal, ' - ', numpy.round(keypoint.pt), ' - ',
                      numpy.round(keypoints[item].pt))
                matchedKeypoints[len(matchedKeypoints) - 1].append(
                    Data(x1=int(round(keypoint.pt[0])), y1=int(round(keypoint.pt[1])),
                         x2=int(round(keypoints[item].pt[0])),
                         y2=int(round(keypoints[item].pt[1]))))
    print('type of matched keypoints - ', type(matchedKeypoints))
    print('minimum distance value - ', minDisVal)
    return matchedKeypoints


def matchClusters(keypointCluster: dict, segs:numpy.uint8, keyMatchPerCluster=1, threshold=40) -> dict:
    """this will match clusters of keypoints with other clusters"""
    matchedSegments = {}
    minimumThresh = 0
    minThreshKeyPair = None
    firstTime = True
    for index1, dictKeyVal1 in enumerate(keypointCluster):  # here we get the segment value that was used as the key
        '''for index2 in range(index + 1,
                                 (len(keypointCluster))):  # here we get the segment value that was used as the key
                                 '''
        for index2, dictKeyVal2 in enumerate(keypointCluster):

            if index2 > index1:  # checking only if the index is pointing forward the current value
                keypointMatchCount = 0
                for keyDes1 in keypointCluster[dictKeyVal1]:
                    for keyDes2 in keypointCluster[dictKeyVal2]:
                        doubleDisVal = scipy.spatial.distance.euclidean(keyDes1.descriptor, keyDes2.descriptor)
                        if doubleDisVal < threshold:  # if under thresh then no problem we have a forged area
                            keypointMatchCount += 1
                            break
                        else:  # trying to get the minimum threshold value and identify the possible forge area
                            if firstTime:
                                minimumThresh = doubleDisVal
                                firstTime = False
                                # append the keypoint descriptor KeyDes object to x value and segment mask value to y of two keypoints
                                minThreshKeyPair = Data(x1=keyDes1,y1=dictKeyVal1, x2=keyDes2, y2=dictKeyVal2)
                            elif doubleDisVal < minimumThresh:
                                minimumThresh = doubleDisVal
                                # append the keypoint descriptor KeyDes object to x value and segment mask value to y of two keypoints
                                minThreshKeyPair = Data(x1=keyDes1, y1=dictKeyVal1, x2=keyDes2, y2=dictKeyVal2)
                # if key point in first patch match with more patches than the threshold number
                if keypointMatchCount >= keyMatchPerCluster:
                    if dictKeyVal1 in matchedSegments:
                        matchedSegments[dictKeyVal1].append(dictKeyVal2)
                    else:
                        matchedSegments[dictKeyVal1] = [dictKeyVal2]

    keyList = [*matchedSegments]
    segmentLength = len(matchedSegments)
    print('lenth of key list - ', len(keyList))
    print('length of segment - ', segmentLength)

    # for index3 in range(segmentLength):
    for num1 in range(0, len(keyList)):
        tempLength = len(matchedSegments)
        if tempLength >= num1:
            if keyList[num1] in matchedSegments:
                for value in matchedSegments[keyList[num1]]:
                    for num2 in range(num1 + 1, len(keyList)):
                        if value in matchedSegments:
                            del matchedSegments[value]

        else:
            break

    # getting the suspect forge area
    if matchedSegments == {}:
        matchedSegments = identifySuspectAreas(minThreshKeyPair=minThreshKeyPair,segments=segs,image=img)

    return matchedSegments


def identifySuspectAreas(minThreshKeyPair:Data, segments:numpy.ndarray, image:numpy.ndarray):
    """this will return array of numpy.ndarray of matched areas"""
    # assuming the sent suspect keypoints are not in the same segment
    rows1, cols1 = numpy.where(segments == minThreshKeyPair.y1)
    rows2, cols2 = numpy.where(segments == minThreshKeyPair.y2)

    startArea1 = min(cols1),min(rows1)  # "startArea1" is a tuple giving x,y value of first segment
    startArea2 = min(cols2), min(rows2)  # "startArea2" is a tuple giving x,y value of second segment
    distribution = (max(cols1) - min(cols1)), (max(rows1) - min(rows1))  # tuple "distribution" gives span of area x,y

    # if (distribution[0] > 8) & (distribution[1] > 8) :  # that is the patch is greater than our HOG window size
    #     for x in range(startArea1,startArea1)

    area1 = image[min(rows1):max(rows1), min(cols1):max(cols1)] # [y1:y2, x1:x2]
    area2 = image[min(rows2):max(rows2), min(cols2):max(cols2)]  # [y1:y2, x1:x2]

    suspectArea = {'area1':area1, 'area2':area2}
    return suspectArea


def calculsteHOGofSuspectAreas(suspectAreas:dict):
    hog = HOGDescriptor()
    returnVAl = hog.compute(img=suspectAreas.area1)
    print(type(returnVAl))


def drawMatchedClusters(image: numpy.uint8, matchedClusters: dict, segments: numpy.ndarray) -> None:
    """this will draw the matched clusters"""
    for matchedClusterNumber in matchedClusters:  # this iteration will give the key value of "matchedClusters" to "matchedClusterNumber"
        rows, cols = numpy.where(segments == matchedClusterNumber)
        for row, col in zip(rows, cols):
            image[row, col] = (255, 255, 255)
        for similarClusterNumber in matchedClusters[matchedClusterNumber]:
            rows, cols = numpy.where(segments == similarClusterNumber)
            for row, col in zip(rows, cols):
                image[row, col] = (255, 255, 255)

    cv2.imshow('clone detected image', image)


def drawClonedArea(image: numpy.uint8, matchedKeypoints: list, segments: numpy.ndarray):
    """this will paint the matched segments"""

    for matchedKeypoint in matchedKeypoints:
        segmentVal = segments[matchedKeypoint.y1][matchedKeypoint.x1]
        rows, cols = numpy.where(segments == segmentVal)
        for row, col in zip(rows, cols):  # zip will stop from the smallest iterable
            image[row, col] = (255, 255, 255)
        print('segment value : ', matchedKeypoint.y2, ' - ', matchedKeypoint.x2)
        segmentVal = segments[matchedKeypoint.y2][matchedKeypoint.x2]
        rows, cols = numpy.where(segments == segmentVal)
        for row, col in zip(rows, cols):  # zip will stop from the smallest iterable
            image[row, col] = (255, 255, 255)

    print("testing segments - ", segments[0][0])
    print("testing matched keypoints - ", matchedKeypoints[0].x1)

    segmentVal = segments[matchedKeypoints[0].x1][matchedKeypoints[0].y1]
    print("testing segment val - ", segmentVal)

    rows, cols = numpy.where(segments == segmentVal)
    # rows = numpy.where((segments == segmentVal).all(axis=1))
    print("length of segments - ", len(segments), "shape - ", segments.shape)  # segments are of the order [y][x]
    print("length of columns - ", len(segments[cols]), "shape - ", segments[cols].shape)
    print("length of rows - ", len(segments[rows]), " shape - ", segments[rows].shape)
    print("length of rows - ", rows.shape)

    cv2.imshow('final image', image)


start_time = time.time()
# img = cv2.imread('/home/waasala/workspace/PycharmProjects/OpenCVBasic/cloningDetection/image.png')
img = resizeImage(cv2.imread('/home/waasala/workspace/gimp/colorFlower-cloned.jpeg'))

keys, des = getSIFTKeyDes(image=img)
segments = getImageSegments(rgbImage=img.copy(), segments=500, sigma=5)
clusturedKeypoints = clusterKeypoints(keypoints=keys, descriptors=des, segments=segments)
matchedClusters = matchClusters(keypointCluster=clusturedKeypoints, keyMatchPerCluster=1, threshold=100)
print('matched clusters - ', matchedClusters)
drawMatchedClusters(image=img, matchedClusters=matchedClusters, segments=segments)
# drawClonedArea(image=img, matchedKeypoints=matchedPositions[:2], segments=segments)
# matchedKeypoints(keypoints=keys, descriptors=des)
# drawSIFTKeysOnly(img)

"""" ############################################ STARTING THREADS ############################ """
'''

class myThread(threading.Thread):
    def __init__(self, threadID: int, name: str, image: numpy.uint8, matchedKeypoints: list, segments: numpy.ndarray):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.image = image
        self.matchedKeypoints = matchedKeypoints
        self.segments = segments

    def run(self):
        print("Starting " + self.name)
        drawClonedArea(image=img, matchedKeypoints=matchedPositions, segments=segments)
        print("Exiting " + self.name)


threads = []

# Create new threads
thread1 = myThread(1, "Thread-1", 1)
thread2 = myThread(2, "Thread-2", 2)

# Start new Threads
thread1.start()
thread2.start()

# Add threads to thread list
threads.append(thread1)
threads.append(thread2)

# Wait for all threads to complete
for t in threads:
    t.join()

print("Exiting Main Thread")
'''
""" ############################################ END OF THREADS ############################ """

print('time of execution - ', time.time() - start_time)

cv2.waitKey(0)
cv2.destroyAllWindows()
