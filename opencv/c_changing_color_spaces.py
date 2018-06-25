
import cv2
import numpy as np
from scipy import ndimage

class ChangeColorSpaces:
    """this class will have functions to change between color spaces (BGR,GREY,HSV )"""

    def get_image(self,path:str)->np.uint8:
        """load an image"""
        image = cv2.imread(path)
        print(image.shape)
        return image

    def show_image(self,windowName:str,image:np.uint8)->None:
        """this will display the images"""
        cv2.imshow(windowName,image)

    def convert2grey(self,image:np.uint8)->np.uint8:
        return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    def convert2HSV(self,image:np.uint8)->np.uint8:
        return cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    def detectObject(self,image:np.uint8)->np.uint8:
        """this will extract a blue colored object(Lennas hair) by following the steps shown below
        convert BGR to HSV
        threshold HSV for range of blue color
        now extract the blue image"""

    #   convert BGR to HSV
        hsvImage = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    #   define range of blue color in HSV
        lower_blue = np.array([90, 60, 90])
        upper_blue = np.array([172, 120, 160])

    #   threshold to get the image in the defined range of HSV converted image
        mask = cv2.inRange(hsvImage,lower_blue,upper_blue)

    #   bitwise AND with mask and original image
        resultImage = cv2.bitwise_and(src1=image,src2=image,mask=mask)

    #   display the images
        cv2.imshow('original image',image)
        cv2.imshow('mask',mask)
        cv2.imshow('result',resultImage)

    def get_HSV_value(self,data:np.uint8=None)->None:
        """this will diaplay the HSV vallue for the BGR values"""
        green = np.uint8([[[0,255,0]]])
        hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
        print(hsv_green)

    def highPassFilter(self,greyImage):
        """this will sharpen the edges/ changre places of the image"""
        kernal_3x3 = np.array([[-1,-1,-1],
                               [-1,8,-1],
                               [-1,-1,-1]])
        k3 = ndimage.convolve(greyImage,kernal_3x3)
        blurred = cv2.GaussianBlur(greyImage,(11,11),0)
        gausian_HPF = k3 - blurred

        cv2.imshow("3x3",k3)
        cv2.imshow("blurred",blurred)
        cv2.imshow("GausianHighPass",gausian_HPF)



class ExecuteColorSpaces:
    """this class will execute the methods in ChangeColorSpaces class"""

    def main(self):
        """the method that need to be called for execution"""
        changeColor = ChangeColorSpaces()
        image = changeColor.get_image('image/Lenna.png')
        # changeColor.detectObject(image=image)
        changeColor.highPassFilter(greyImage=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
        # image = changeColor.get_image('image/colorPuppy_1.jpg')
        # changeColor.show_image('original image',image)
        #
        # changeColor.show_image('grey image',changeColor.convert2grey(image))
        # changeColor.show_image('HEV image', changeColor.convert2HSV(image))
        # changeColor.detectObject(image)
        # changeColor.get_HSV_value()

        cv2.waitKey(0)
        cv2.destroyAllWindows()
