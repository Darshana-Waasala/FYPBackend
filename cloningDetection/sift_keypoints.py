import numpy
import cv2
import scipy.spatial.distance
from matplotlib import pyplot as plt


def resizeImage(image: numpy.uint8):
    """this will resize image if dimensions are greter than 512"""
    if image.shape[0] > 512:
        width = int(numpy.around((image.shape[1]) / 2))
        height = int(numpy.around(image.shape[0] / 3))
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


def getSIFTKeysOnly(image: numpy.uint8) -> None:
    """this will calculate only the SIFT keypoints only"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)
    img = cv2.drawKeypoints(gray, kp, image)
    cv2.imshow('sift_keypoints.jpg', img)
    return None


def matchedKeypoints(keypoints: list, descriptors: numpy.ndarray, threshold=0.0004) -> list:
    """this function will return the matched within a threshold"""
    for index, keypoint in enumerate(keypoints):  # index is starting from ZERO
        for item in range(index + 1, (len(keypoints))):  # item is starting from 1
            doubleDisVal = scipy.spatial.distance.euclidean(descriptors[index], descriptors[item])
            if doubleDisVal < threshold:
                print(index, ' - ', doubleDisVal, ' - ', numpy.round(keypoint.pt), ' - ',
                      numpy.round(keypoints[item].pt))


def testSIFT():
    # img1 = cv2.imread('/home/waasala/workspace/gimp/test1.jpeg', 0)  # queryImage
    img2 = resizeImage(cv2.imread('/home/waasala/workspace/gimp/colorFlower_rotated.jpeg', 0))  # trainImage
    # img1 = resizeImage(cv2.imread('/home/waasala/workspace/gimp/colorFlower_rotated.jpeg', 0))[10:200,70:350]  # query
    img1 = img2.copy()[160:180, 150:170]

    '''remove the query area form the image'''
    img2[160:180, 150:170]  = 0
    # cv2.imshow('testing', img1)
    # cv2.waitKey(0)
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, img2, flags=2)
    plt.imshow(img3), plt.show()


testSIFT()
'''img2 = resizeImage(cv2.imread('/home/waasala/workspace/gimp/colorFlower_rotated.jpeg', 0))
# img2 = resizeImage(cv2.imread('/home/waasala/workspace/gimp/colorFlower_rotated.jpeg', 0))[10:200,70:200]
img2[160:180, 150:170] = 0
# img2[0:290, 0:300, 1] = 0
# img2[0:290, 0:300, 2] = 0
print(img2.shape)
cv2.imshow('check this out', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
# img = cv2.imread('/home/waasala/workspace/PycharmProjects/OpenCVBasic/cloningDetection/image.png')
#
# keys, des = getSIFTKeyDes(image=img)
# matchedKeypoints(keys, des)
# getSIFTKeysOnly(img)'''


cv2.waitKey(0)
cv2.destroyAllWindows()
