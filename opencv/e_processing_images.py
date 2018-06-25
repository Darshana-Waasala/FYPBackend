import cv2
import numpy
from cv2.cv2 import HOGDescriptor
from scipy import ndimage
import scipy.spatial.distance


class ProcessImages:
    """this will have the code segments of the chapter 3 of reference book"""

    def readImage(self, path) -> numpy.uint8:
        """function to read image"""
        return cv2.imread(path);

    def showImage(self, name: str, image: numpy.uint8) -> None:
        cv2.imshow(name, image);
        return None

    def resizeImage(self, image: numpy.uint8) -> numpy.uint8:
        """this will resize the image if it is larger than 512*512"""
        print(image.shape);
        if image.shape[0] > 512:
            width = int(numpy.around((image.shape[1]) / 2))
            height = int(numpy.around(image.shape[0] / 2))
            resizedImage = cv2.resize(src=image, dsize=(width, height))
            return resizedImage
        return image

    def highPassFilter(self, image: numpy.uint8) -> None:
        """sharpen the image using two methods 1-using a kernal 2-subtracting the image from the blurred"""
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
        cv2.imshow('test', image)
        kernal_3x3 = numpy.array([[-1, -1, -1],
                                  [-1, 8, -1],
                                  [-1, -1, -1]])
        sharpenImgUsingKernal = ndimage.convolve(input=image, weights=kernal_3x3);
        cv2.imshow("sharpened image using kernal", sharpenImgUsingKernal);

        blurredImage = cv2.GaussianBlur(src=image, ksize=(11, 11), sigmaX=0)
        sharpnedImage = image - blurredImage
        cv2.imshow('sharpened using image reduction', sharpnedImage)
        return None

    def findEdges(self, image: numpy.uint8) -> None:
        """this function will find the edges of an image using medianBlur and Laplacian kernals"""
        bluredImage = cv2.medianBlur(src=image, ksize=7)
        cv2.imshow('blured image', bluredImage)
        greyImage = cv2.cvtColor(src=bluredImage, code=cv2.COLOR_BGR2GRAY)
        laplacianImage = cv2.Laplacian(src=greyImage, ddepth=cv2.CV_8U, ksize=5)
        cv2.imshow('laplacian image', laplacianImage)
        normalizedInverseAlpha = (1 / 255) * (255 - laplacianImage)
        channels = cv2.split(m=laplacianImage)
        i = 0
        for channel in channels:
            channel[:] = channel * normalizedInverseAlpha
            print(i)
            i += 1
        finalImage = cv2.merge(mv=channels)
        cv2.imshow('the final image', finalImage)
        return None

    def findEdgesWithCranny(self, image: numpy.uint8) -> None:
        """this function will detect the edges of the passed image using canny edge detector"""
        edgeDetectedImae = cv2.Canny(image=image, threshold1=150, threshold2=200)
        cv2.imshow('edge detected image using canny', edgeDetectedImae)
        return None

    def squareContourDetection(self) -> None:
        """this function will detect the drawn square contours"""
        image = numpy.zeros(shape=(200, 200), dtype=numpy.uint8)
        image[50:150, 50:150] = 255
        image = cv2.imread(filename="image/Lenna.png", flags=0)

        ret, thresh = cv2.threshold(src=image, thresh=127, maxval=255, type=0)
        image, contours, hierachy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        color = cv2.cvtColor(src=image, code=cv2.COLOR_GRAY2BGR)
        resultImage = cv2.drawContours(image=color.copy(), contours=contours, contourIdx=-1, color=(0, 255, 0),
                                       thickness=2)
        cv2.imshow('result image', resultImage)
        return None

    def contourStep2(self) -> None:
        """this will find the bounding box, minimum area circle, minimum enclosing circle of an object"""
        image = cv2.pyrDown(src=cv2.imread(filename='image/contour.png', flags=cv2.IMREAD_UNCHANGED))
        greyImage = cv2.cvtColor(src=image.copy(), code=cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(src=greyImage, thresh=127, maxval=255, type=cv2.THRESH_BINARY)
        image2, contours, hierachy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL,
                                                      method=cv2.CHAIN_APPROX_SIMPLE)
        i = 0

        for contour in contours:
            # find bounding box cordinates
            x, y, w, h = cv2.boundingRect(points=contour)
            cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=1)

            # find minimum area
            rect = cv2.minAreaRect(points=contour)
            # calculate coordinates of minimum area rectangle
            box = cv2.boxPoints(box=rect)
            # normalize cordinates to integers
            box = numpy.int0(box)
            # draw contours
            cv2.drawContours(image=image, contours=[box], contourIdx=0, color=(0, 0, 255), thickness=1)

            # calculate center and radius and minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(points=contour)
            # cast to integers
            center = (int(x), int(y))
            radius = int(radius)
            # draw the circle
            cv2.circle(img=image, center=center, radius=radius, color=(255, 0, 0), thickness=1)

        # cv2.drawContours(image=image,contours=contours,contourIdx= -1,color=(255,0,0),thickness=1)
        cv2.imshow('contours', image)
        return None

    def lineDetection(self, image: numpy.uint8) -> None:
        """this will detect line using two methods of Hough transformation functions
        HoughLines and HoughLinesP"""

        greyImage = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
        edgesMarked = cv2.Canny(image=greyImage, threshold1=50, threshold2=120)

        minLineLength = 10
        maxLineGap = 5
        lines = cv2.HoughLinesP(image=edgesMarked, rho=1, theta=numpy.pi / 180, threshold=50, lines=10,
                                minLineLength=minLineLength, maxLineGap=maxLineGap)
        print('no of lines detected', lines)

        # if lines != None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                print(x1)
                cv2.line(img=image, pt1=(x1, y1), pt2=(x2, y2), color=255, thickness=1)

        cv2.imshow('edges', edgesMarked)
        cv2.imshow('lines', image)

    def copyPasteROI(self, image):
        """this will copy one image part to another place"""
        roi = image[40:270, 350:540]  # [y1:y2, x1:x2]
        image[0:230, 0:190] = roi
        cv2.imshow('new image', image)
        cv2.imwrite('cloningDetection/image.png', image)


def calculateHOGofSuspectAreas():
    print('ok....')
    hog = HOGDescriptor((8, 8), (8, 8), (8, 8), (8, 8), 9)  # 16x32 -> col,row -> x,y
    # hogDescriptor1 = hog.compute(img=cv2.imread('/home/waasala/workspace/gimp/colorFlower.jpeg')[0:128, 0:64])
    # hogDescriptor2 = hog.compute(img=cv2.imread('/home/waasala/workspace/gimp/colorFlower.jpeg')[0:128, 0:64])
    image = ProcessImages.resizeImage(0,cv2.imread('/home/waasala/workspace/gimp/colorBird.jpeg'))#imread('/home/waasala/workspace/PycharmProjects/OpenCVBasic/cloningDetection/image.png'))
    hogDescriptor1 = hog.compute(img=image[90:240, 100:300])
    hogDescriptor2 = hog.compute(img=image[70:220, 300:500])
    # # hogDescriptor1 = hog.detect(img=cv2.imread('/home/waasala/workspace/gimp/colorFlower.jpeg')[350:550, 350:550])
    # # hogDescriptor2 = hog.detect(img=cv2.imread('/home/waasala/workspace/gimp/colorFlower.jpeg')[250:550, 250:550])
    print('type of hog descriptor_1 - ', type(hogDescriptor1), ' first value of descriptor_1 - ', hogDescriptor1[0])
    print('type of hog descriptor_2 - ', type(hogDescriptor2), ' first value of descriptor_2 - ', hogDescriptor2[0])
    print('distance : ', scipy.spatial.distance.euclidean(hogDescriptor1, hogDescriptor2))
    image[90:240, 100:300,0] =0
    image[55:205, 300:500,1] =0
    cv2.imshow('test', hogDescriptor1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


calculateHOGofSuspectAreas()


class ExecuteProcessImages:
    """this will execute the ProcessImages class"""

    def main(self) -> None:
        """main method that need to be called from outside"""
        # image = ProcessImages.readImage(self,path="image/Lenna.png")
        # ProcessImages.showImage(self,'testing',image)
        # ProcessImages.highPassFilter(self,image)
        # ProcessImages.findEdges(self,image)
        # ProcessImages.findEdgesWithCranny(self,image)
        # ProcessImages.squareContourDetection(self)
        # ProcessImages.contourStep2(self)
        # ProcessImages.lineDetection(self,image=ProcessImages.resizeImage(self,cv2.imread('image/colorBird.jpeg')))
        # cv2.imshow('resized image',ProcessImages.resizeImage(self,cv2.imread('image/colorBird.jpeg')))
        ProcessImages.copyPasteROI(self, ProcessImages.resizeImage(self, cv2.imread('image/colorBird1.jpeg')))

        cv2.waitKey(0)
        cv2.destroyAllWindows()
