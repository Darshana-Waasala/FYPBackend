
import cv2
import numpy as np

class ArithmeticOperations:
    """this class will deal with the arithmetic operations on the images"""

    def get_image(self,path:str)->np.uint8:
        """this method will load the image"""
        image = cv2.imread(path)
        print(image.shape)
        return image

    def show_image(self,name:str,image:np.uint8)->None:
        """this will display the image"""
        cv2.imshow(name,image)

    def image_addition_intuition(self)->None:
        """this shows that the image addition will saturate at 255 and not go beyond that"""
        x = np.uint8([250])
        y = np.uint8([10])
        print(cv2.add(x,y)) # this will only give 255 NOT 260

    def image_blending(self, img1:np.uint8, img2:np.uint8)->np.uint8:
        """blending is also a kind of image addition.
        but we can add with some weights of the image. it is like
        (1-a)f(x) + (a)f(x) + b"""
        # to do blending we need to have images of same size
        return cv2.addWeighted(img1,0.7,img2,0.3,0)

    def bitwise_operations(self,img1:np.uint8,img2:np.uint8)->np.uint8:
        """this method will do copy and paste a non rectangular area of image"""

    # it is required to put the image to the top left corner. so created a ROI
        rows,cols,channels = img2.shape
        roi = img1[0:rows, 0:cols]

    # we'll create a mask of logo and create it's inverse mask also
        img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg, img2_fg)
        img1[0:rows, 0:cols] = dst

        return img1


class ArithmeticOperatorExecutor:
    """this class will instantiate ArithmeticOperations class and execute it's methods
    this is the class that will be called from the main method"""

    a_operations = ArithmeticOperations()
    # image = a_operations.get_image('image/Lenna.png')
    # a_operations.show_image('lenna',image)

    # a_operations.image_addition_intuition()
    # blended_image = a_operations.image_blending(a_operations.get_image('image/Lenna.png'),a_operations.get_image('image/triangle.png'))
    # a_operations.show_image('blended image',blended_image)

    a_operations.show_image('edited image',a_operations.bitwise_operations(cv2.imread('image/colorPuppy_1.jpg'),cv2.imread('image/Lenna.png')))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
