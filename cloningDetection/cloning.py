# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage import img_as_ubyte
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt

import cv2
import numpy as np


image = img_as_float(io.imread('Lenna.png'))
# image = img_as_float(io.imread('horses_copy.png'))

numSegments = 100
# get the segments
# apply SLIC and extract (approximately) the supplied number
# of segments
segments = slic(image, n_segments=numSegments, sigma=5)

# boundries marked image
segmented_image = mark_boundaries(image, segments)

# converting the image to opencv comptible
segmented_image = segmented_image[:,:,::-1] # converted to bgr order
cv_image = img_as_ubyte(segmented_image) #convert to cv type image

# marking key points
gray= cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(cv_image,None)
# kp2, desc = sift.detectAndCompute(cv_image,None)
# print('key point',type(kp2))
# print('key point',kp2.shape)
# print('descriptor',type(desc))

cv_image = cv2.drawKeypoints(gray,kp,cv_image)
cv2.imshow('sift_keypoints.jpg',cv_image)



cv2.waitKeyEx(0)
cv2.destroyAllWindows()