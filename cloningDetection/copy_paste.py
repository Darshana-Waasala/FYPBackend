import cv2
from matplotlib import pyplot as plt

x1 = 100
y1 = 100

x2 = 200
y2 = 200

new_x = 50
new_y = 50

# image = cv2.imread('../image/Lenna.png')
# roi = image[y1:y2,x1:x2]
# image[new_y:new_y+(y2-y1),new_x:new_x+(x2-x1)] = roi
#
# cv2.imshow('image',image)
# cv2.imwrite('../image/copyLenna.png',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

sift = cv2.xfeatures2d.SIFT_create()
greyImage = cv2.imread('../image/Lenna.png',0)
keypoints1, descriptor1 = sift.detectAndCompute(greyImage,None)
keypoints2, descriptor2 = sift.detectAndCompute(greyImage,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptor1,descriptor2, k=2)

img3 = cv2.imread('../image/Lenna.png',0)
img3 = cv2.drawMatchesKnn(greyImage,keypoints1,greyImage,keypoints2,matches,img3,flags=2)
plt.imshow(img3),plt.show()


