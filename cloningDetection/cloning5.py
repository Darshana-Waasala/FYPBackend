"""this is to do cloning detection using ORB and brute force matching"""
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

def getORBKesDes(image:numpy.ndarray, segments:numpy.ndarray):
    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    keys, des = orb.detectAndCompute(image=image,mask=None)

    '''separate the keypoints into two groups to use them in the knn clustering'''
    splitedDescriptions = numpy.array_split(des, 2)
    desRange1 = len(splitedDescriptions[0])

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    print('type of keys = ', len(keys))
    print('shape of descriptors = ', des.shape)
    print('length of descriptors = ', len(des[0]))

    # Match descriptors.
    matches = bf.match(splitedDescriptions[0], splitedDescriptions[1])
    '''the parameters -> query feature, train feature, cluster size'''
    print('length of matches : ', len(matches), ' type of matches - ', type(matches))

    # Apply ratio test
    count = 0
    bestMatches = {}  # dictionary to hold the cluster matches
    good = []
    for m in matches:
        count += 1
        print('match distance : ', m.distance, ' match train index : ',m.trainIdx, ' match query index : ', m.queryIdx, ' train image index : ', m.imgIdx)
        """if m.distance < 0.75 * n.distance:
            good.append([m])

            count += 1
            # print('distance : ', (m.distance / n.distance), m.trainIdx, n.trainIdx)
            print('n distance - ', n.distance, ' m distance - ', m.distance, ' m query index = ', m.queryIdx,
                  ' n train index = ', n.trainIdx)
            x1 = int(round(keys[m.queryIdx].pt[0]))
            y1 = int(round(keys[m.queryIdx].pt[1]))
            x2 = int(round(keys[n.trainIdx + desRange1 - 1].pt[0]))
            y2 = int(round(keys[n.trainIdx + desRange1 - 1].pt[1]))

            # img[int(round(keys[m.queryIdx].pt[1]) - 9):int(round(keys[m.queryIdx].pt[1] + 9)),
            #     int(round(keys[m.queryIdx].pt[0] - 9)):int(round(keys[m.queryIdx].pt[0] + 9)), 1] = 0

            # img[int(round(keys[n.trainIdx + desRange1 - 1].pt[1]) - 9):int(
            #     round(keys[n.trainIdx + desRange1 - 1].pt[1] + 9)),
            # int(round(keys[n.trainIdx + desRange1 - 1].pt[0] - 9)):int(
            #     round(keys[n.trainIdx + desRange1 - 1].pt[0] + 9)), 2] = 0

            '''draw patches around the keypoints matched'''
            img[y1 - 9:y1 + 9, x1 - 9:x1 + 9, 1] = 0
            img[y2 - 9:y2 + 9, x2 - 9:x2 + 9, 1] = 0

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
                bestMatches[segmentVal1] = [segmentVal2]"""

        print('total count : ', count)

        return bestMatches


'''################################## execution start position ##############################################'''

start_time = time.time()
# img = cv2.imread('/home/waasala/workspace/PycharmProjects/OpenCVBasic/cloningDetection/image.png')
img = cv2.imread('/home/waasala/workspace/gimp/colorFlower-cloned.jpeg')
# img = (cv2.imread('/home/waasala/workspace/gimp/00007tamp4.jpg'))
# img = cv2.imread('/home/waasala/workspace/gimp/P1000472tamp5.jpg')
# img = cv2.imread('/home/waasala/workspace/gimp/P1000293tamp9.jpg')

segments = getImageSegments(rgbImage=img,segments=500,sigma=5)
matchedSegmetns = getORBKesDes(image=img,segments=segments)



print('time of execution - ', time.time() - start_time)

cv2.waitKey(0)
cv2.destroyAllWindows()

"""general knowledge
in python opencv image point can be accessed as -> img[10,100] = (255,255,255) # x=100,y=10
                                this is similar to -> img[x,y] = (255,255,255)"""