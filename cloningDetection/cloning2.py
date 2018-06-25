
# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

import cv2
import numpy

# imagePath = '../image/Lenna.png'
imagePath = '../image/copyLenna.png'
imageFloat = img_as_float(io.imread(imagePath))
segments = slic(imageFloat, n_segments=7, sigma=5)
segmentedImage = mark_boundaries(imageFloat,segments)

# converting the image to opencv comptible
segmented_image = segmentedImage[:,:,::-1] # converted to bgr order
cv_image = img_as_ubyte(segmented_image) #convert to cv type image

grey = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(grey,None)
print('first ',type(descriptor[0]))
print('first ',type(descriptor))

# classifiedKeypoints = [0]*len(numpy.unique(segments))
classifiedKeypoints=[]
classifiedDescriptors=[]
for i in range(0,len(numpy.unique(segments))):
    classifiedKeypoints.append([])
    classifiedDescriptors.append([])

# print(classifiedKeypoints)
# to catogirize the keypoints
i=0;
for keypoint in keypoints:
    keypoint_x = int(numpy.round(keypoint.pt[0]))
    keypoint_y = int(numpy.round(keypoint.pt[1]))

    classifiedKeypoints[segments[keypoint_x][keypoint_y]].append(keypoint)
    classifiedDescriptors[segments[keypoint_x][keypoint_y]].append(descriptor[i])
    i+=1

print('classified',type(classifiedDescriptors[0]))
print('classified',classifiedDescriptors[0])

# match the keypoints of different segments
# BFMatcher with default params
bf = cv2.BFMatcher()
finalImage = numpy.zeros((cv_image.shape[0],cv_image.shape[1],3), numpy.uint8)
for i in range(0,classifiedKeypoints.__len__()-1):
    for ii in range(i,classifiedKeypoints.__len__()-1):
        matches = bf.knnMatch(classifiedDescriptors[i],classifiedDescriptors[ii], 2)
        finalImage = cv2.drawMatchesKnn(cv_image,classifiedKeypoints[i],cv_image,classifiedKeypoints[ii],matches,finalImage,flags=2)

print('keypoints',len(keypoints))
print('descriptors',descriptor[0])

# finalImage = cv2.imread(imagePath)
# cv_image = cv2.drawKeypoints(image=cv_image,keypoints=keypoints,outImage=cv_image,color=(255, 255, 255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# image = cv2.imread(imagePath)
# cv2.imshow('testing',cv_image)
cv2.imshow('final image',finalImage)
cv2.waitKey(0)
cv2.destroyAllWindows()