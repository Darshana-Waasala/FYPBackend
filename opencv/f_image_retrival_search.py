import cv2
import numpy
from cv2.cv2 import HOGDescriptor


def cornerdetection(image: numpy.uint8) -> None:
    """this will detect the corners of the image using CornerHarris algorithm"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = numpy.float32(gray)
    dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=23, k=0.04)
    print(type(dst))
    print(dst[0])
    image[dst > 0.01 * dst.max()] = [0, 255, 0]
    cv2.imshow('corners', image)
    return None


def siftKeypointDetection(image: numpy.uint8) -> None:
    """this funciton will detect keypoints using SIFT algorithm"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptor = sift.detectAndCompute(gray, None)
    print(type(keypoints))
    print(descriptor[0].dtype)
    print(type(keypoints[0]))
    # img = cv2.drawKeypoints(image=image, outImage=image, keypoints=keypoints,
    #                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(51, 163, 236))
    # cv2.imshow('sift_keypoints', img)
    return None


def surfKeypointDetection(image: numpy.uint8) -> None:
    """this funciton will detect keypoints using SURF algorithm"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create(1000)
    keypoints, descriptor = surf.detectAndCompute(gray, None)
    print(type(keypoints))
    print(type(descriptor))
    print(type(keypoints[0]))
    img = cv2.drawKeypoints(image=image, outImage=image, keypoints=keypoints,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(51, 163, 236))
    cv2.imshow('sift_keypoints', img)
    return None


def flannBasedMach(image: numpy.uint8) -> None:
    """this will detect image keypoints using SIFT algorithm and match them using flannBasedMatch"""

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image, None)
    print('length of keypoints - ', len(kp1))

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des1, k=2)
    print('length of matches : ', len(matches), ' type of matches - ', type(matches))

    # ratio test as per Lowe's paper
    count = 0;
    for i, (m, n) in enumerate(matches):
        if abs(m.distance - n.distance) < 40:
            count += count
            print('distance : ', (n.distance - m.distance), m.trainIdx, n.trainIdx)
            # print(n.distance - m.distance)  # n has the higher value than m (always - did not encounter other way round)




def flannBasedMachHOG(image: numpy.uint8) -> None:
    """this will detect image keypoints using SIFT algorithm and match them using flannBasedMatch"""

    hog = HOGDescriptor((8, 8), (8, 8), (8, 8), (8, 8), 9)  # 16x32 -> col,row -> x,y

    hogDescriptor1 =[]
    hogDescriptor2 =[]
    for i in range(1,20):  # iterating to get 10 descriptors
        hogDescriptor1.append(hog.compute(img=image[90-i:240-i, 100:300]))
        hogDescriptor2.append(hog.compute(img=image[70+i:220+i, 300:500]))

    # converting arrays to numpy.ndarray of data type float32
    print('length of all descriptors - ', len(hogDescriptor2))
    print('length - ', len(hogDescriptor2[1]))
    print('length - ', len(hogDescriptor1[2]))
    print('type - ', type(numpy.float32(hogDescriptor1)))
    hogDescriptor1 = numpy.float32(hogDescriptor1)
    hogDescriptor2 = numpy.float32(hogDescriptor2)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(hogDescriptor1, hogDescriptor2, k=2)
    print('length of matches : ', len(matches), ' type of matches - ', type(matches))

    # ratio test as per Lowe's paper
    count = 0;
    for i, (m, n) in enumerate(matches):
        if abs(m.distance - n.distance) < 0.8:
            count += 1
            print('distance : ', (n.distance - m.distance), m.trainIdx, n.trainIdx)
            # print(n.distance - m.distance)  # n has the higher value than m (always - did not encounter other way round)
    print('total count of matches : ',count)


# image = cv2.imread('/home/waasala/workspace/PycharmProjects/OpenCVBasic/cloningDetection/image.png')
image = cv2.imread('/home/waasala/workspace/gimp/colorFlower-cloned.jpeg')
# cornerdetection(image)
# siftKeypointDetection(image=image)
# surfKeypointDetection(image=image)
flannBasedMach(image)
# flannBasedMachHOG(image=image)
cv2.waitKey(0)
cv2.destroyAllWindows()
