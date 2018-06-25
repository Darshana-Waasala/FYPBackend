# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

import cv2
import numpy as np

# load the image and convert it to a floating point data type
image = img_as_float(io.imread('/home/waasala/workspace/PycharmProjects/OpenCVBasic/image/Lenna.png')) # ../image/Lenna.png

# loop over the number of segments
# for numSegments in (50 , 100, 400):
#     # apply SLIC and extract (approximately) the supplied number
#     # of segments
#     segments = slic(image, n_segments=numSegments, sigma=5)
#     print(segments)
#     print(len(np.unique(segments)))
#
#
#     # show the output of SLIC
#     fig = plt.figure("Superpixels -- %d segments" % (numSegments))
#     ax = fig.add_subplot(1, 1, 1)
#     ax.imshow(mark_boundaries(image, segments))
#     plt.axis("off")
segments = slic(image, n_segments=500, sigma=5)
print(segments[100])
print(segments[101])
print(segments[102])
print(segments[103])
print(type(segments))
print(len(segments))
print(len(np.unique(segments)))

# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (50))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments,color=(1,1,1,)))
plt.axis("off")

# show the plots
plt.show()

