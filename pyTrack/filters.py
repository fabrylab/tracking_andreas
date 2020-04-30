import numpy as np
from skimage.filters import gaussian
from skimage.morphology import remove_small_objects
from skimage.morphology import binary_erosion, binary_dilation

def remove_large_objects(mask, max_size):
    mask[remove_small_objects(mask, max_size)] = False
    return mask


### image filters

def filter_dog(img, s1=0, s2=0):
    return gaussian(img, sigma=s1) - gaussian(img, sigma=s2)



def normalize(image, lb=0.1, ub=99.9):
    '''
    nomralizies image to  a range from 0 and 1 and cuts of extrem values
    e.g. lower tehn 0.1 percentile and higher then 99.9m percentile

    :param image:
    :param lb: percentile of lower bound for filter
    :param ub: percentile of upper bound for filter
    :return:
    '''

    image = image - np.percentile(image, lb)  # 1 Percentile
    image = image / np.percentile(image, ub)  # norm to 99 Percentile
    image[image < 0] = 0.0
    image[image > 1] = 1.0
    return image


### mask filters( filters after detection

def fitler_objects_size(mask, min_size=None, max_size=None):

    if isinstance(min_size,(int,float)):
        mask=remove_small_objects(mask,min_size=min_size)
    if isinstance(max_size,(int,float)):
        mask=remove_large_objects(mask,max_size=max_size)
    return mask

def mask_cleanup1(mask, min_size=None, max_size=None):
    mask = binary_erosion(mask)
    mask = binary_dilation(mask)
    if isinstance(min_size,(int,float)):
        mask=remove_small_objects(mask,min_size=min_size)
    if isinstance(max_size,(int,float)):
        mask=remove_large_objects(mask,max_size=max_size)
    return mask