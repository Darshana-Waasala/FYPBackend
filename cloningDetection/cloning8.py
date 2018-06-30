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
    def __init__(self, query_keys: list, query_des: numpy.ndarray, train_keys: numpy.ndarray, train_des: numpy.ndarray):
        """this is the constuctor of the method"""
        self.query_keys = query_keys
        self.query_des = query_des
        self.train_keys = train_keys
        self.train_des = train_des


class Channels:
    def __init__(self, red: int, green: int, blue: int):
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
    print(totalSegments)
    return totalSegments
    # return 50


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
    if image.shape[0] > 1100:
        width = int(numpy.around((image.shape[1]) / 2))
        height = int(numpy.around(image.shape[0] / 2))
        print('height , width:',width,',',height)
        resize_image = cv2.resize(src=image, dsize=(width, height))
        return resizeImage(resize_image)
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


def getLocalizedImageSegment(segmentValue: int, segments: numpy.ndarray, image: numpy.ndarray) -> (
        numpy.ndarray, numpy.ndarray):
    """this will get the approximated image segment for the given segment mask value and
    the corresponding whole image with removed segment"""

    '''assuming the sent suspect key points are not in the same segment'''
    '''numpy.where will give the list of rows and list of cols corresponding each pixel in the segment 
    with the specified value'''
    rows, cols = numpy.where(segments == segmentValue)

    '''image ROI is taken as image[y1:y2,x1:x2]'''
    roi = image[min(rows):max(rows), min(cols):max(cols)]

    '''removing the roi from the full image'''
    imageWithRemovedROI = image.copy()
    imageWithRemovedROI[min(rows):max(rows), min(cols):max(cols)] = 0

    return roi, imageWithRemovedROI


def getArrangedDictionary(segments: numpy.ndarray, image: numpy.ndarray) -> dict:
    """this will prepare a dictionary of key ponts and their descriptors of each approximated patches"""
    sift = cv2.xfeatures2d.SIFT_create()

    '''prepare the dictionary that is to be returned and is arranged as,
    dict{segmentValue:FeatureObject,....}'''
    dictionary = {}

    '''get the unique values of the segmentation mask'''
    unique_segment_values = numpy.unique(segments)

    '''iterate over the segment to get the localized sift keypoints'''
    for segmentValue in unique_segment_values:
        local_image_segment, image_without_roi = getLocalizedImageSegment(segmentValue=segmentValue, segments=segments,
                                                                          image=image)

        '''# for testing purposes
        local_img_path = '/home/waasala/remove/' + str(segmentValue) + '_li.jpeg'
        image_without_roi_path = '/home/waasala/remove/' + str(segmentValue) + '_fi.jpeg'
        cv2.imwrite(str(local_img_path), local_image_segment)
        cv2.imwrite(str(image_without_roi_path), image_without_roi)'''

        query_keys, query_des = sift.detectAndCompute(cv2.cvtColor(local_image_segment,cv2.COLOR_BGR2GRAY), None)
        train_keys, train_des = sift.detectAndCompute(cv2.cvtColor(image_without_roi,cv2.COLOR_BGR2GRAY), None)

        '''adding only if at least one key point is found'''
        if query_des is not None:
            dictionary[segmentValue] = Features(query_keys=query_keys, query_des=query_des, train_keys=train_keys,
                                                train_des=train_des)

    return dictionary


def getMatchedPatches(arranged_dictionary: dict, segments: numpy.ndarray, key_matches_per_cluster=5,
                      cluster_matches_per_cluster=2) -> dict:
    """this function will iterate over the localized descriptors and return the dictionary of matched items"""
    # leastDistance = 5000

    '''prepare the BF matcher for the knn match between the localized keypoints'''
    bf = cv2.BFMatcher()  # BFMatcher with default params

    matched_segments = {}  # the dictionary to hold the matched segments

    '''iterate over the arranged patches to get (nxn - n!) iterations'''
    for segmentValue in arranged_dictionary:
        dic_value = arranged_dictionary[segmentValue]
        '''to make sure that we check the segment combinations that have not met before'''
        condition = dic_value.query_des is not None
        if condition:
            matches = bf.knnMatch(queryDescriptors=dic_value.query_des, trainDescriptors=dic_value.train_des, k=2)
            count = 0  # to keep the count of the matches under the threshold
            matching_segments = set()
            '''iterating over the matches to filter out those under the required threshold'''
            for i, m in enumerate(matches):

                '''since some of the matches do not get both n and m'''
                if len(m) > 1:
                    if m[0].distance < 0.4 * m[1].distance:
                        '''ALTERNATIVES TRIED FOR THE ABOVE IF CONDITION
                        if m[1].distance != 0:
                            if (m[0].distance / m[1].distance) < leastDistance:
                                leastDistance = (m[0].distance / m[1].distance)'''

                        '''now we have found a good match'''
                        count += 1
                        '''get the segment value of the matched key point'''
                        col, row = dic_value.train_keys[m[0].trainIdx].pt  # key.pt -> (x,y)|(col,row)|(width,height)

                        # print('train col,row:', int(round(col)), ',', int(round(row)), 'query seg value:', segmentValue,
                        #       '|len(train_keys):', len(dic_value.train_keys), '|len(query_keys):',
                        #       len(dic_value.query_keys), ' |n trainIdx:', m[1].trainIdx, ' |n queryIdx:', m[1].queryIdx,
                        #       ' |m trainIdx', m[0].trainIdx, ' |m queryIdx', m[0].queryIdx)

                        matching_segments.add(segments[int(round(row)), int(round(col))])

            '''filling the segment into the dictionary if there are required number of matches in the patches'''
            if (count >= key_matches_per_cluster) & (len(matching_segments) >= cluster_matches_per_cluster):
                print('matching...', segmentValue, ',', matching_segments)
                if segmentValue in matched_segments:
                    matched_segments[segmentValue].extend(
                        x for x in matching_segments if x not in matched_segments[segmentValue])
                # elif matching_segment in matched_segments:
                #     matched_segments[matching_segment].append(segmentValue)
                else:
                    matched_segments[segmentValue] = matching_segments

        else:
            continue

    # print('least distance rechorded .......', leastDistance)
    return matched_segments


def getArrangedDictionaryORB(segments: numpy.ndarray, image: numpy.ndarray) -> dict:
    """this will prepare a dictionary of key ponts and their descriptors of each approximated patches"""

    # create orb detector
    orb = cv2.ORB_create()

    '''prepare the dictionary that is to be returned and is arranged as,
    dict{segmentValue:FeatureObject,....}'''
    dictionary = {}

    '''get the unique values of the segmentation mask'''
    unique_segment_values = numpy.unique(segments)

    '''iterate over the segment to get the localized sift keypoints'''
    for segmentValue in unique_segment_values:
        local_image_segment, image_without_roi = getLocalizedImageSegment(segmentValue=segmentValue, segments=segments,
                                                                          image=image)

        '''# for testing purposes
        local_img_path = '/home/waasala/remove/' + str(segmentValue) + '_li.jpeg'
        image_without_roi_path = '/home/waasala/remove/' + str(segmentValue) + '_fi.jpeg'
        cv2.imwrite(str(local_img_path), local_image_segment)
        cv2.imwrite(str(image_without_roi_path), image_without_roi)'''

        query_keys, query_des = orb.detectAndCompute(image=local_image_segment, mask=None)
        train_keys, train_des = orb.detectAndCompute(image=image_without_roi, mask=None)

        '''adding only if at least one key point is found'''
        if query_des is not None:
            dictionary[segmentValue] = Features(query_keys=query_keys, query_des=query_des, train_keys=train_keys,
                                                train_des=train_des)

    return dictionary


def getMatchedPatchesORB(arranged_dictionary: dict, segments: numpy.ndarray, key_matches_per_cluster=1,
                         cluster_matches_per_cluster=3) -> dict:
    """this function will iterate over the localized descriptors and return the dictionary of matched items"""
    least_distance = 5000

    '''prepare the BF matcher for the knn match between the localized keypoints'''
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # BFMatcher with default params

    matched_segments = {}  # the dictionary to hold the matched segments

    '''iterate over the arranged patches to get (nxn - n!) iterations'''
    for segmentValue in arranged_dictionary:
        dic_value = arranged_dictionary[segmentValue]
        '''to make sure that we check the segment combinations that have not met before'''
        condition = dic_value.query_des is not None
        if condition:
            matches = bf.knnMatch(queryDescriptors=dic_value.query_des, trainDescriptors=dic_value.train_des, k=2)
            count = 0  # to keep the count of the matches under the threshold
            matching_segments = set()
            '''iterating over the matches to filter out those under the required threshold'''
            for i, m in enumerate(matches):

                '''since some of the matches do not get both n and m'''
                if len(m) > 1:
                    print('m/m[0] trainIdx:', m[0].trainIdx, 'm/m[0] queryIdx:', m[0].queryIdx, ' | n/m[1] trainIdx :',
                          m[1].trainIdx, ' | n/m[1] queryIdx :', m[1].queryIdx)
                    if m[1].distance != 0:
                        if (m[0].distance/m[1].distance) < least_distance:
                            least_distance = m[0].distance/m[1].distance
                    if m[0].distance < 0.75 * m[1].distance:
                        '''ALTERNATIVES TRIED FOR THE ABOVE IF CONDITION
                        if m[1].distance != 0:
                            if (m[0].distance / m[1].distance) < least_distance:
                                least_distance = (m[0].distance / m[1].distance)'''

                        '''now we have found a good match'''
                        count += 1
                        '''get the segment value of the matched key point'''
                        col, row = dic_value.train_keys[m[0].trainIdx].pt  # key.pt -> (x,y)|(col,row)|(width,height)

                        # print('train col,row:', int(round(col)), ',', int(round(row)), 'query seg value:', segmentValue,
                        #       '|len(train_keys):', len(dic_value.train_keys), '|len(query_keys):',
                        #       len(dic_value.query_keys), ' |n trainIdx:', m[1].trainIdx, ' |n queryIdx:', m[1].queryIdx,
                        #       ' |m trainIdx', m[0].trainIdx, ' |m queryIdx', m[0].queryIdx)

                        matching_segments.add(segments[int(round(row)), int(round(col))])

            '''filling the segment into the dictionary if there are required number of matches in the patches'''
            if (count >= key_matches_per_cluster) & (len(matching_segments) >= cluster_matches_per_cluster):
                print('matching...', segmentValue, ',', matching_segments)
                if segmentValue in matched_segments:
                    matched_segments[segmentValue].extend(
                        x for x in matching_segments if x not in matched_segments[segmentValue])
                # elif matching_segment in matched_segments:
                #     matched_segments[matching_segment].append(segmentValue)
                else:
                    matched_segments[segmentValue] = matching_segments

        else:
            continue

    print('least distance rechorded .......', least_distance)
    return matched_segments


def drawMatchedClusters(image: numpy.ndarray, matchedClusters: dict, segments: numpy.ndarray) -> None:
    """this will draw the matched clusters"""
    for matchedClusterNumber in matchedClusters:
        print('matched parent cluster:', matchedClusterNumber, ' |children :', end='', flush=True)

        rows1, cols1 = numpy.where(segments == matchedClusterNumber)
        for row1, col1 in zip(rows1, cols1):
            image[row1, col1, 0] = 0  # (255, 255, 255)

        if len(matchedClusters[matchedClusterNumber]) >= 1:
            for similarClusterNumber in matchedClusters[matchedClusterNumber]:
                rows2, cols2 = numpy.where(segments == similarClusterNumber)
                print(similarClusterNumber, ',', end='', flush=True)

                ''' Draw a diagonal blue line with thickness of 5 px parameters: pt1 is in (x,y) order '''
                cv2.line(img=image, pt1=(cols1[0], rows1[0]), pt2=(cols2[0], rows2[0]), color=(255, 0, 0), thickness=2)

                for row2, col2 in zip(rows2, cols2):
                    image[row2, col2, 1] = 0  # (255, 255, 255)
        print('\n')

    cv2.imshow('clone detected image', (image))


def getEnlargeLocalImageSegment(matchedPatches: dict, segments: numpy.ndarray, image: numpy.ndarray):
    """this will enlarge the image patch based on averageColor
    this function will return a """

    '''get the image dimensions'''
    img_height, img_width, img_channels = image.shape

    '''obtaining the unique segment values'''
    uniqueSegmentValues = numpy.unique(segments)

    '''iterating over the segment values to get the individual pixel values'''
    for segmentValue in uniqueSegmentValues:
        rows, cols = numpy.unique(segments == segmentValue)

        for row in rows:
            for col in cols:

                '''confirm segment does not lie in  4 boundaries of the image'''
                if row != 0:  # if the pixel is not in the left corner

                    '''get the image segment value next to the pixel'''
                    tempegVal = segments[row - 1, col]


def getAverageImageColorValues(image: numpy.ndarray, segment: numpy.ndarray) -> dict:
    """this will generate average color value per segment and return the result as a dictionary with
    segment value as key and average pixel value as value"""
    segment_average_color_value = {}
    unique_segment_values = numpy.unique(segment)

    '''iterating over segment values to get the average corresponding image color values'''
    for segmentValue in unique_segment_values:
        temp_red_value = 0
        temp_green_value = 0
        temp_blue_value = 0
        rows, cols = numpy.where(segment == segmentValue)

        '''iterating over the pixels of the segment to get the average of three channels'''
        for row, col in zip(rows, cols):
            temp_red_value = temp_red_value + image[row, col, 0]
            temp_green_value = temp_green_value + image[row, col, 1]
            temp_blue_value = temp_blue_value + image[row, col, 2]

        segment_average_color_value[segmentValue] = Channels(red=int(round(temp_red_value / len(rows))),
                                                             green=int(round(temp_green_value / len(rows))),
                                                             blue=int(round(temp_blue_value / len(rows))))

    #     '''iterating over the segment to color the image'''
    #     for row, col in zip(rows, cols):
    #         image[row, col, 0] = segment_average_color_value[segmentValue].red
    #         image[row, col, 1] = segment_average_color_value[segmentValue].green
    #         image[row, col, 2] = segment_average_color_value[segmentValue].blue
    #
    # cv2.imshow('new colored image', resizeImage(image=image))
    return segment_average_color_value


""" ################################## execution start position ############################################## """
start_time = time.time()
# img = resizeImage(cv2.imread('/home/waasala/Education/Level 4_Semester two theory/group project/data/MICC_F600/central_park.png'))
img = resizeImage(cv2.imread('/home/waasala/workspace/gimp/00007tamp4.jpg'))

# ''' using SIFT descriptor with brute force KNN match
segs = getImageSegments(rgbImage=img, segments=getMostAppropriteSegementNumber(image=img), sigma=5)
print('got segments :', len(numpy.unique(segs)))
arrangedDict = getArrangedDictionary(segments=segs, image=img)
print('arranged dictionary', len(arrangedDict))
matchedPatches = getMatchedPatches(arranged_dictionary=arrangedDict, segments=segs, key_matches_per_cluster=2,
                                   cluster_matches_per_cluster=2)
print('matched patches:', len(matchedPatches))
drawMatchedClusters(image=img, matchedClusters=matchedPatches, segments=segs)
# getAverageImageColorValues(image=img, segment=segments)

print('time of execution - ', time.time() - start_time)
cv2.waitKey(0)
cv2.destroyAllWindows()
# '''

''' using ORB descriptor with brute force KNN match
segs = getImageSegments(rgbImage=img, segments=getMostAppropriteSegementNumber(image=img), sigma=5)
print('got segments :', len(numpy.unique(segs)))
arrangedDict = getArrangedDictionaryORB(segments=segs, image=img)
print('arranged dictionary', len(arrangedDict))
matchedPatches = getMatchedPatchesORB(arranged_dictionary=arrangedDict, segments=segs, key_matches_per_cluster=2,
                                      cluster_matches_per_cluster=2)
print('matched patches:', len(matchedPatches))
drawMatchedClusters(image=img, matchedClusters=matchedPatches, segments=segs)
# getAverageImageColorValues(image=img, segment=segments)

print('time of execution - ', time.time() - start_time)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
