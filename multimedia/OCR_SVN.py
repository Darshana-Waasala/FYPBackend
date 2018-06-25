import cv2
import numpy

def deskew(image)->numpy.uint8:
    """this is to de-skew the image"""
    m = cv2.moments(array=image)
    if abs(m['mu02']) < 1e-2:
        return image.copy()
    skew = m['mu11'] / m['mu02']
    M = numpy.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(image, M, (SZ, SZ), flags=affine_flags)
    print('type of : ',type(m))
    print(m)
    return img

def bowTrainer():
    cv2.k

image = cv2.imread('/home/waasala/workspace/PycharmProjects/OpenCVBasic/image/Lenna.png')
grey = cv2.cvtColor(src=image,code=cv2.COLOR_BGR2GRAY)
# deskew(grey) # this one is giving errors

cv2.waitKey(0)
cv2.destroyAllWindows()
