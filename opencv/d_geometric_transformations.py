
import cv2
import numpy as np

class GeometricTransformations:
    """this class will have the methods for different geometric transformations"""

    def get_image(self,path:str)->np.uint8:
        return cv2.imread(path)

    def show_image(self,windowName:str,image:np.uint8)->np.uint8:
        cv2.imshow(windowName,image)


class ExecuteTransformationMethods:
    """this class will execute the transformations defined in the GeometricTransformation class"""

    transformations = GeometricTransformations()
    image = transformations.get_image('image/Lenna.png')
    transformations.show_image('original image',image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
