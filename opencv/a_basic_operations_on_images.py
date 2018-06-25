
import cv2
import numpy as np

# declare a global variable
# image_path = ''

class ImageDetails:
    """class for the image details passing"""
    def __init__(self,size,shape,datatype,):
        self.size = size
        self.shape = shape
        self.dataType = datatype


class BasicOperations:
    """contains all the basic functions"""

    def __init__(self, path:str):
        """class constructor"""
        self.image_path = path # using a instance variable

    def get_image(self):
        """get image instance"""
        image = cv2.imread(self.image_path)
        print(image.shape)
        return image

    def show_image(self, windowName:str, image:np.uint8)->None:
        """view the image"""
        cv2.imshow(windowName, image)

    def test(self, image:np.uint8)->None:
        """does some basic testing"""
        pixel = image[0,0]
        print('pixel at 100,100 : ', pixel)
        # colors are in the order BGR
        print('blue value at 100,100 : ',image[100,100,0])
        print('converting the pixel at 1000,100 to white')
        image[100,100] = [255,255,255]
        print(image[100,100])

    def get_blue_image(self, image:np.uint8)->np.uint8:
        """seperate image to individual colors"""
        imageDetails = image.shape # shape is in the order ROWS,COLS,CHANNELS
        row=0;col=0
        new_image = np.zeros((imageDetails[0], imageDetails[1], 3), np.uint8)
        print('just created the new image')
        while row < imageDetails[0]:
            col = 0
            while col < imageDetails[1]:
                new_image[row,col] =[image[row,col,0],0,0]
                col = col + 1
            row = row +1
        print('just finished the while loop')
        return new_image

    def get_grey_image(self, image:np.uint8)->np.uint8:
        """to get the grey of the colord image"""
        greyImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        return greyImage

    def get_red_shade(self, image:np.uint8)->np.uint8:
        """easy way to get the image components"""
        image[:,:,0] = 0 # remove blue shade
        image[:,:,1] = 0 # remove green shade
        return image

    def get_image_details(self, image)->ImageDetails:
        """will return all the information of the image"""
        return ImageDetails(image.size,image.shape,image.dtype)

    def image_roi(self, image:np.uint8)->np.uint8:
        """copy and paste - cloning of the Region of Interest"""
        lamp = image[130:225,40:100] #[y1:y2, x1:x2]
        image[0:95,40:100] = lamp
        return image

    def splliting_merging(self, image)->None:
        """testing for image splliting to individual channels and merging to get the original image"""
        b,g,r = cv2.split(image)
        cv2.imshow('blue',b)
        cv2.imshow('green',g)
        cv2.imshow('red',r)
        mergedImg = cv2.merge((b,g,r))
        cv2.imshow('merged image',mergedImg)

    def make_border(self,image:np.uint8)->np.uint8:
        """draw  a border around the image"""
        replicate = cv2.copyMakeBorder(image,30,30,30,30,cv2.BORDER_REPLICATE)
        reflect = cv2.copyMakeBorder(image, 30, 30, 30, 30, cv2.BORDER_REFLECT)
        reflect101 = cv2.copyMakeBorder(image, 30, 30, 30, 30, cv2.BORDER_REFLECT_101)
        wrap = cv2.copyMakeBorder(image, 30, 30, 30, 30, cv2.BORDER_WRAP)
        constant = cv2.copyMakeBorder(image, 30, 30, 30, 30, cv2.BORDER_CONSTANT)

        cv2.imshow('original',image)
        cv2.imshow('replicate',replicate )
        cv2.imshow('reflect',reflect )
        cv2.imshow('reflect101',reflect101 )
        cv2.imshow('wrap',wrap )
        cv2.imshow('constant',constant )


class ClassExecuter:
    """execute the class methods"""
    class_instance = BasicOperations('image/test.jpg')
    image = class_instance.get_image()
    # class_instance.show_image('image', image)
    # blue_image = class_instance.get_blue_image(image)
    # class_instance.show_image('blue_image',blue_image)
    # class_instance.show_image('gery image',class_instance.get_grey_image(image))
    # print('image size: ',class_instance.get_image_details(image).size,'\nimage shape: ',
    #       class_instance.get_image_details(image).shape,'\nimage data type: ',class_instance.get_image_details(image).dataType)
    # class_instance.show_image("unknown",class_instance.get_red_shade(image))
    # class_instance.show_image('cloned',class_instance.image_roi(image))
    # class_instance.splliting_merging(image)
    # class_instance.test(image)
    class_instance.make_border(image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
