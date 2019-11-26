
from typing import Optional, Callable, Tuple, List, NoReturn
from functools import partial

import matplotlib.pyplot as plt
#import matplotlib.image as img

import numpy as np
import cv2 as cv
import PIL as pil

def binarise(src: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
    """
        Take a grayscale image of any range : 
            [0.0, 1.0]  (float)
            [0, 255]    (int)

        And binarise it, specifying a threshold or using the 
        image intensity mean value as default.

        The binarisation occurs as follows

        pixels_intensity >= threshold --> _max
        pixels_intensity <  threshold --> _min

        '_min' and '_max' are inferred from the source image's dtype

        dtype ~ float :
            _min = 0.0
            _max = 1.0

        dtype ~ int :
            _min = 0
            _max = 255
        
        Arguments:
                  src : A grayscale image, of type numpy.ndarray
            threshold : An optional value to specify the 
                        'cutoff' intensity value.
                        Defaults to mean intensity value.

        Returns : 
                  dst : A binary image, of type numpy.ndarray
    """
    dst = src.copy()
    
    # Assign correct max and min, according to the source image dtype.
    _floats    = [np.float, np.float16, np.float32, np.float64, np.float128]
    _iffloat   = partial(
        lambda im, f_val, i_val: f_val if im.dtype in _floats else i_val, 
        src
    )
    
    _max, _min = list(map(_iffloat, [1.0, 0.0], [255, 0]))
    
    if threshold:
        dst[ dst >= threshold ] = _max
        dst[ dst <  threshold ] = _min
    else:
        threshold = src.mean()  # Default value for binarization
        dst[ dst >= threshold ] = _max
        dst[ dst <  threshold ] = _min
    
    return dst
##

def rescale_img(src: np.ndarray) -> np.ndarray:
    """
    """
    return src / 255.0
##

def reverse(src: np.ndarray) -> np.ndarray:
    """
    """
    return 1.0 - src
#

def complement(src: np.ndarray) -> np.ndarray:
    """
    """
    return 1.0 - src
#

def side_by_side(
    image1: np.ndarray, 
    image2: np.ndarray, 
    title1: Optional[str] = None, 
    title2: Optional[str] = None,
    _figsize: Optional[Tuple[int]] = (15, 10),
    **kw
) -> NoReturn: 
    """
        Show two matplotlib.pyplot.imshow images, side by side.
        
        Optional arguments : 
            Title for each subplot.
            Figsize, specified as a tuple.
    """
    
    fig = plt.figure(figsize = _figsize)
    
    fig.add_subplot(1, 2, 1)
    if len(image1.shape) == 3:
        plt.imshow(image1)
    else:
        plt.imshow(image1, cmap = 'gray')
    if title1:
        plt.title(title1, size = 18)
    
    fig.add_subplot(1, 2, 2)
    if len(image2.shape) == 3:
        plt.imshow(image2)
    else:
        plt.imshow(image2, cmap = 'gray')
    if title2:
        plt.title(title2, size = 18)
##

def structuring_circle(radius: int, size: Optional[int] = None):
    ''' 
        radius : Radius of circle inside the 2D ndarray which will be filled with ones.
        size   : Optional size of the rectangle 2D ndarray which will contain the circle. 
        Inspired from : 
            https://stackoverflow.com/questions/53326570/how-to-create-sphere-inside-a-ndarray-python
    '''
    if size:
        assert size >= 2*radius, 'Circle overflows matrix surface !'
        assert size % 2 == 0, 'Size must be even !'
    else:
        size = 2*radius
        
    A = np.zeros((size+1, size+1))
    AA = A.copy() 
    D = AA.copy()
    
    ''' (x0, y0) : coordinates of center of circle inside A. '''
    x0, y0 = int(np.floor(A.shape[0]/2)), int(np.floor(A.shape[1]/2))


    for x in range(x0-radius, x0+radius+1):
        for y in range(y0-radius, y0+radius+1):
            ''' deb: measures how far a coordinate in A is far from the center. 
                deb>=0: inside the sphere.
                deb<0: outside the sphere.'''   
            deb = radius - abs(x0-x) - abs(y0-y) 
            D[x, y] = deb
            if (deb)>=0: AA[x,y] = 1
    
    AA = np.uint8(AA)
    
    return AA
##

def opening(src: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
        As defined in pages 644 and 645 :
            'The opening A by B is the erosion of A by B, 
            followed by a dilation of the result by B'
        
        This function is should be equivalent to :
            cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    """
    return cv.dilate(cv.erode(src, kernel, iterations=iterations), kernel, iterations=iterations)
##

def closing(src: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
        As defined in pages 644 and 6 45 :
            'The closing of A by B is simply the dilation of A by B, 
            followed by erosion of the result by B.'
        
        This function is should be equivalent to :
            cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    """
    return cv.erode(cv.dilate(src, kernel, iterations=iterations), kernel, iterations=iterations)
##

