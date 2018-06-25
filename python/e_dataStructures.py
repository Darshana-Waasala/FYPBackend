import timeit
import numpy

class DataStructures:

    def CreateArrays(self)->None:
        print('creating an array with ones of length 4 - ', numpy.ones(4))
        print('creating an array from 0 to 3 - ', numpy.arange(0,4))
        return None

    def BasicOperations(self)->None:
        testArray = numpy.ones(4)
        print('adding one to each array item - ', testArray +1)
        test2DArray = numpy.ones((3,3))
        print('muliplying 2D array by 2 - ',test2DArray*2)

        a = numpy.array([1, 2, 3, 4])
        b= numpy.array([4, 2, 2, 4])
        print('checking for individual element similarity - ',a==b)
        print('checking for individual element value factor - ', a > b)
        print('checking array similarity - ',numpy.array_equal(a,b))
        return None

    def BasicReductions(self)->None:
        a = numpy.array([1, 2, 3, 4])
        b = numpy.arange(1,5)
        print('two arrays are equal(indiviaual testing) - ',numpy.equal(a,b))
        print('two arrays are equal(as a whole) - ',numpy.array_equal(a,b))
        print('sum of the arrays -> (method 1) - ',numpy.sum(a),'(method 2) - ',a.sum())
        print('basic item ', a[2:3])


# DataStructures.CreateArrays(4)
# DataStructures.BasicOperations(0)
DataStructures.BasicReductions(0)