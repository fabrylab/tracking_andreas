from skimage.filters import threshold_otsu
import numpy as np

def segementation_sd(image, f=5, mask_area=None, min_treshold=None, max_treshold=None):
    '''
    segmentation based on distance from mean in terms of stadart deviations
    :param image: image(or  part of an image)
    :param f: factor of how many standart deviations from the mean the threshold will be set
    :return:
    '''

    if isinstance(mask_area, np.ndarray):
        thres = np.mean(image[mask_area]) + f * np.std(image[mask_area])
    else:
        thres = np.mean(image) + f * np.std(image)
    if min_treshold:
        if min_treshold > thres:  # take min_treshold as new threshold
            thres = min_treshold
    if max_treshold:
        if max_treshold < thres:  # take max_treshold as new threshold
            thres = max_treshold

    mask = image > thres
    if isinstance(mask_area,np.ndarray):
        mask[~mask_area] = False
    return mask, thres


def segmentation_fixed_threshold(image, mask_area=None, thres=None):
    mask = image > thres
    if isinstance(mask_area,np.ndarray):
        mask[~mask_area] = False
    return mask, thres


def segmentation_otsu(img, mask_area=None):
    if isinstance(mask_area, np.ndarray):
        thres = threshold_otsu(img[mask_area])
    else:
        thres = threshold_otsu(img)
    mask = img > thres
    if isinstance(mask_area,np.ndarray):
        mask[~mask_area] = False
    return mask, thres