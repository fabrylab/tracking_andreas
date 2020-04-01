

from skimage.measure import label as measure_label
from skimage.feature import peak_local_max
from skimage.filters import gaussian,threshold_otsu,sobel
from skimage.morphology import watershed,remove_small_objects
from skimage.measure import regionprops
import cv2 as cv
from scipy.stats import gaussian_kde
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation
import numpy as np
import copy

from pyTrack.utilities import *


def cdb_add_detection(frame, db, detect_funct, cdb_types=["detections"], layer=None, detect_type = None, image=None):
    '''
    wrapping writing and reading to database
    :param frame:
    :param detect_funct:
    :param layer:
    :param detect_type:
    :param image:
    :return:
    '''
    if not isinstance(image, db.table_image):
        cdb_image = db.getImage(frame=frame, layer=1)
    else:
        cdb_image = image
    if not isinstance(image, np.ndarray):
        if not isinstance(image,  db.table_image):
            img = cdb_image.data.astype(float)
    else:
        img = image
    detection, mask = detect_funct(img, db)  # must return as iterable



    # setting markers at the weighted centroids of cells in the cdb files
    try:
        if detect_type=="diff":
            for detect, types in zip(detection, cdb_types):
                db.setMarkers(frame=frame, x=detect[:, 1], y=detect[:, 0], type=types)
        else:
            db.setMarkers(frame=frame, x=detection[:, 1], y=detection[:, 0], type=cdb_types)


    except Exception as e:
        print("Error", e)

    try:
        if detect_type == "diff":
            m = (mask[0] * 1 + mask[1] * 2).astype("uint8")
        else:
            m = mask.astype("uint8")
        db.setMask(image=cdb_image, data=m)
    except Exception as e:
        print("Error", e)


def detect_diff(image,  *args, **kwargs):

    masks = segemtation_diff_img(image)
    positive_mask, negative_mask = masks
    mask= positive_mask*2+negative_mask
    mask_cdb = mask.astype(np.uint8)

       # labeling each sell and calcualting various properties of the labeld regions
    labeled_pos = measure_label(positive_mask)
    regions_pos = regionprops(labeled_pos, intensity_image=image)
    detections_pos=[r.weighted_centroid for r in regions_pos]
    detections_pos = np.array(detections_pos)

    labeled_neg = measure_label(negative_mask)
    regions_neg = regionprops(labeled_neg, intensity_image=image)
    detections_neg = [r.weighted_centroid for r in regions_neg]
    detections_neg = np.array(detections_neg)

    return (detections_pos, detections_neg), masks




def segemtation_diff_img(diff_blurr):
    med = np.median(diff_blurr)
    thresh1 = threshold_otsu(diff_blurr[diff_blurr < med])
    thresh2 = threshold_otsu(diff_blurr[diff_blurr > med])
    positive_mask=diff_blurr<thresh1
    negative_mask=diff_blurr>thresh2

    positive_mask= remove_small_objects(positive_mask, min_size=100)
    negative_mask = remove_small_objects(negative_mask, min_size=100)

    return positive_mask,negative_mask







def detect1(image, db, detailed_analysis=True,threshold="otsu"):

    ##returns and uses weighted centroid
    '''
    parameters:
    image- a cdb image object, refering to a maximum projection image
    db- a cdb database object
    detailed_analysis- boolean, specifies if segmentation masks should be save to the cdb file

    This function performes segmentation of cells on maximum Projection images. It uses a band pass filter (only allowing
    medium sized, round shaped objects) consisting of two gaussian filters and a threshold caclulated from mean and empirical
    standard deviation. Detection of cells are calculated as the weighted centroid of segmented cells, if they are more
    then 75 pixel from the edge and are not much smaller the the average segmented cell.
    '''

    # image segmentation by a broad band filter
    img = image.data.astype(float)
    img2 = gaussian(img, 15)-gaussian(img,30)
    mu, std = np.mean(np.ravel(img2)), np.std(np.ravel(img2), ddof=1)
    if threshold=="otsu":
        mask = img2 > threshold_otsu(img2)
    if threshold=="mean_std":
        mask = img2 > mu + 5 * std


    # optional saving of the masks to
    if detailed_analysis:
        mask_cdb = mask.astype(np.uint8)
        db.setMask(data=mask_cdb, image=image)

    # labeling each sell and calcualting various properties of the labeld regions
    labeled = measure_label(mask)
    regions = regionprops(labeled, intensity_image=img2)



    # creating a list of all areas of relevant cells
    areas = []
    for r in regions:
        y, x = r.weighted_centroid
        if (75 < x < img.shape[1] - 75) and (75 < y < img.shape[0] - 75):
            areas.append(r.area)

    # exclusion of all objects that are unusually small and that are to close to the edge of the image
    mu = np.mean(areas)
    std = np.std(areas, ddof=1)
    detections = []
    for r in regions:
        y, x = r.weighted_centroid
        if (75 < x < img.shape[1] - 75) and (75 < y < img.shape[0] - 75) and (r.area > mu - 1.5*std):
            detections.append((x, y))
    detections = np.array(detections)

    # setting markers at the weighted centroids of cells in the cdb files
    try:
        db.setMarkers(image=image, x=detections[:, 0], y=detections[:, 1], type='detection_prelim')
    except Exception as e:
        print("Error", e)

    return detections


def detect2(image, db, detailed_analysis=True, threshold="otsu"):
    ##returns  normal centroid
    '''
    parameters:
    image- a cdb image object, refering to a maximum projection image
    db- a cdb database object
    detailed_analysis- boolean, specifies if segmentation masks should be save to the cdb file

    This function performes segmentation of cells on maximum Projection images. It uses a band pass filter (only allowing
    medium sized, round shaped objects) consisting of two gaussian filters and a threshold caclulated from mean and empirical
    standard deviation. Detection of cells are calculated as the weighted centroid of segmented cells, if they are more
    then 75 pixel from the edge and are not much smaller the the average segmented cell.
    '''

    # image segmentation by a broad band filter
    img = image.data.astype(float)
    img2 = gaussian(img, 15) / gaussian(img, 30)
    mu, std = np.mean(np.ravel(img2)), np.std(np.ravel(img2), ddof=1)
    if threshold == "otsu":
        mask = img2 > threshold_otsu(img2)
    if threshold == "mean_std":
        mask = img2 > mu + 5 * std

    # optional saving of the masks to
    if detailed_analysis:
        mask_cdb = mask.astype(np.uint8)
        db.setMask(data=mask_cdb, image=image)

    # labeling each sell and calcualting various properties of the labeld regions
    labeled = measure_label(mask)
    regions = regionprops(labeled, intensity_image=img2)

    # creating a list of all areas of relevant cells
    areas = []
    for r in regions:
        y, x = r.centroid
        if (75 < x < img.shape[1] - 75) and (75 < y < img.shape[0] - 75):
            areas.append(r.area)

    # exclusion of all objects that are unusually small and that are to close to the edge of the image
    mu = np.mean(areas)
    std = np.std(areas, ddof=1)
    detections = []
    for r in regions:
        y, x = r.centroid
        if (75 < x < img.shape[1] - 75) and (75 < y < img.shape[0] - 75) and (r.area > mu - 1.5 * std):
            detections.append((x, y))
    detections = np.array(detections)

    # setting markers at the weighted centroids of cells in the cdb files
    try:
        db.setMarkers(image=image, x=detections[:, 0], y=detections[:, 1], type='detection_prelim')
    except Exception as e:
        print("Error", e)

    return detections



def detect_fl(image, db, detailed_analysis=True, threshold="loose",spacing_factor=1):
    ### uses fluorescent images
    ##returns and uses weighted centroid
    '''
    parameters:
    image- a cdb image object, refering to a maximum projection image
    db- a cdb database object
    detailed_analysis- boolean, specifies if segmentation masks should be save to the cdb file

    This function performes segmentation of cells on maximum Projection images. It uses a band pass filter (only allowing
    medium sized, round shaped objects) consisting of two gaussian filters and a threshold caclulated from mean and empirical
    standard deviation. Detection of cells are calculated as the weighted centroid of segmented cells, if they are more
    then 75 pixel from the edge and are not much smaller the the average segmented cell.
    '''

    # image segmentation by a broad band filter

    # parameters:

   # spacing_factor = 1  # factor to make threshold more restrictive
    size = 10000  # sampling size of pixels for threshold calculation
    min_cell_size = 200  # minimal accepted cell size in pixel
    prominence_decision = 0.90  # prominence(more like a rough estimation) level between two maxima that decides weather
    # both will be regarded as correct. Calculated by minimal "hight" (distance to background) on straight line between the
    # two maxima devided by "hight" of the smaller maximum
    max_distance = 5  # maximal distance allowed between two maxima
    kde_factor = 2

    img = image.data.astype(float)
    img2 = gaussian(img, 1) - gaussian(img, 30)  ## usefull for background noise (but not really necessary)
    mu, std = np.mean(np.ravel(img2)), np.std(np.ravel(img2), ddof=1)
    if threshold == "otsu":
        otsu_thresh = threshold_otsu(img2)
        mask = img2 > otsu_thresh
    if threshold == "mean_std":
        mask = img2 > mu + 1 * std
    if threshold == "loose":

        flat_img2 = img2.flatten()
        # using only small subset of pixel values to reduce calclualtion cost

        subset = np.random.choice(flat_img2, size, replace=False)
        min_value = np.min(subset)
        max_value = np.max(subset)
        xs = np.linspace(min_value, mu + 5 * std,
                         size)  # maximum value is "far beyond" intensities that are expected for the background
        # kerel density estimation to smoot the histogramm
        kde_sample_scott = gaussian_kde(subset, bw_method= size**(-1./(1+4))*kde_factor) # # this is scotts method with additional factor
        kd_estimation_values = kde_sample_scott(xs)  # getting values of the kde

        # calculating first and second derivative with rough approximation
        first_derivative = kd_estimation_values[:-1] - kd_estimation_values[1:]
        second_derivative = first_derivative[:-1] - first_derivative[1:]

        ### strategie: finding maximum of second derivative, will always be just at the bottom of a guass distributed curve
        max_pos_1 = np.where(kd_estimation_values == np.max(kd_estimation_values))[0][
            0]  ## maximum of histogramm (not actually needed)
        max_pos = xs[max_pos_1]
        thresh_1 = np.where(second_derivative == np.max(second_derivative[max_pos_1:]))[0]  # maximum of second derivative
        thresh = xs[thresh_1]  # threshold in terms of intensity
        thresh_new = max_pos + np.abs(
            max_pos - thresh) * spacing_factor  # usung absolute distance to maximum to effectively use spacing factor
        # spacing_factor: factor to get wider spacing, will generally be set to 1
        mask = binary_fill_holes(img2 > thresh_new)
    # segmentation with this threshold and filling holes

    labeled_objects = measure_label(mask)  # labeling
    labeled_objects_filtered = remove_small_objects(labeled_objects, min_size=min_cell_size)  # filtering

    # distance transform and finding local maxima
    mask3 = labeled_objects_filtered > 0
    distance = distance_transform_edt(mask3)

    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((20, 20)),
                                labels=mask3, threshold_abs=np.sqrt(min_cell_size / 4))    # absolute threshold, especiall y nice to exclude cells with
        # with highly irregular edges
    markers = measure_label(binary_dilation(local_maxi,
                                            iterations=1))  # dilation to connect all regions with 3 pixel distance ## could use diffrent aproache or selection a little later
    maxim_coords = [x.centroid for x in
                    regionprops(markers)]  # centroid gives one point for maxima that have a flat peak
    maxim_coords = [[int(x), int(y)] for x, y in maxim_coords]  ## getting coordinates of centroid as integers

    # finding "prominence"sharten höhe" of maxima
    # wrtinging dictionary with keys: label of obejct where maximu is found
    maxima_list = {key: [] for key in np.unique(labeled_objects_filtered)}  # initializing keys for all lable values
    maxima_list.pop(0, None)  # remove zero (zero would be background)

    # inserting coordinates of maxima
    for coord in maxim_coords:  ## note some values at the edge are empty, but thats ok
        lable_value = labeled_objects_filtered[
            coord[0], coord[1]]  ## values are not in order because of filtering of small objects
        if lable_value != 0:
            maxima_list[lable_value].append(coord)


    # filtering for distance between maxima ()
    maxima_list_corrected = copy.deepcopy(maxima_list)
    for keys, values in maxima_list.items():
        l=len(values)
        if l == 0:  # removing some empty entrances (happens when object is close to edge of the image)
            maxima_list_corrected.pop(keys, None)
        if l == 0:  # removing some empty entrances (happens when object is close to edge of the image)
            maxima_list_corrected.pop(keys, None)
        if l >1 :  # only looking at objects with two maxima inside tem

            ## removing all maxima that are to close
            while True:
                comp_matrix=claculate_comp_matrix_distance(values)
                pixel_dist=np.nanmin(comp_matrix[:,:])
                if pixel_dist >= max_distance or np.isnan(pixel_dist):
                    break
                index=np.where(comp_matrix == pixel_dist)
                ## removing to values and finding mean place between them

                max1=values[index[0][0]]
                max2=values[index[1][0]]
                mean_max=[int(np.mean(pair)) for pair in zip(max1,max2)]
                values[index[0][0]]=mean_max #replacing first maximum
                values.pop(index[1][0]) #removing second maximum



            ## removing all maxima that are not prominent enough:
            while True:
                comp_matrix = claculate_comp_matrix_prominence(values,distance)
                min_prominence = np.nanmax(comp_matrix[:, :])   ### "minimal" prominence is like 1 and "maximum" is 0
                if min_prominence <=  prominence_decision or np.isnan( min_prominence):
                    break
                index = np.where(comp_matrix == min_prominence)
                ## removing to values and finding mean place between them
                max1 = values[index[0][0]]
                max2 = values[index[1][0]]
                mean_max = [int(np.mean(pair)) for pair in zip(max1, max2)]
                values[index[0][0]] = mean_max  # replacing first maximum
                values.pop(index[1][0])  # removing second maximum

        maxima_list_corrected[keys] = values

    # producing labeled objects out of maxima list again
    corected_maxima = np.zeros(np.shape(labeled_objects_filtered))
    maxima_list_corrected_flat = [x for y in maxima_list_corrected.values() for x in y]
    for maxima in maxima_list_corrected_flat:
        corected_maxima[maxima[0], maxima[1]] = 1
    markers = measure_label(corected_maxima)

    # watershedding from new maxima
    labels = watershed(-distance, markers, mask=mask3)

    # optional saving of the masks to
    if detailed_analysis:
        mask_cdb = mask3.astype(np.uint8)
        db.setMask(data=mask_cdb, image=image)
    regions = regionprops(labels, intensity_image=img2)

    # exclusion of all objects that are to close to the edge of the image
    detections = []
    for r in regions:
        y, x = r.centroid
        if (75 < x < img.shape[1] - 75) and (75 < y < img.shape[0] - 75):
            detections.append((x, y))
    detections = np.array(detections)

    # setting markers at the centroids of cells in the cdb files
    try:
        db.setMarkers(image=image, x=detections[:, 0], y=detections[:, 1], type='detection_prelim')
    except Exception as e:
        print("Error", e)
    return detections


def detect_maxProj1(image, threshold="otsu"):
    # returns weighted centroid
    '''
    parameters:
    image- numpy array, a maximum projection image
    returns:
    detections- array, object weighted centers, [object,xy_coordinates]
    coords- list, list of coordinates that cover the whole object area
    labeled- labeled mask of segmentation


    Detection and segmentation of cells from maximum projection images.This function uses a band pass filter (only
    allowing medium sized, round shaped objects) consisting of two gaussian filters and a threshold caclulated from mean
    and empirical standard deviation. Detection of cells are calculated as the weighted centroid of segmented cells, if
    they are more then 75 pixel from the edge and are not much smaller the the average segmented cell.

    '''


    # segmentation of cells by broadband filter and threshold form stadart devaition and mean
    img = image.astype(float)
    img2 = gaussian(img, 15) / gaussian(img, 30)
    mu, std = np.mean(np.ravel(img2)), np.std(np.ravel(img2), ddof=1)
    if threshold == "otsu":
        mask = img2 > threshold_otsu(img2)
    if threshold == "mean_std":
        mask = img2 > mu + 5 * std

    labeled = measure_label(mask)
    regions = regionprops(labeled, intensity_image=img2) # calculates various properties of the segmented objects




    # filtering out objects that are unusually small and lie close to the image edge
    areas = []
    for r in regions:
        y, x = r.weighted_centroid
        if (75 < x < img.shape[1] - 75) and (75 < y < img.shape[0] - 75):
            areas.append(r.area)
    mu = np.mean(areas)  # mean and sd of all object areas not close to the image edges
    std = np.std(areas, ddof=1)
    detections = []
    coords = []
    for r in regions:
        y, x = r.weighted_centroid
        if (75 < x < img.shape[1] - 75) and (75 < y < img.shape[0] - 75) and (r.area > mu - 1.5*std): # filtering
            detections.append((x, y))  # stores x,y coordinates of object center (weighted by gauss-filtered image)
            coords.append(r.coords)  # stores slice of object positions
    detections = np.array(detections)

    return detections, labeled, coords



def detect_maxProj2(image, threshold="otsu"):

    # returns normal centroid
    '''
    parameters:
    image- numpy array, a maximum projection image
    returns:
    detections- array, object weighted centers, [object,xy_coordinates]
    coords- list, list of coordinates that cover the whole object area
    labeled- labeled mask of segmentation


    Detection and segmentation of cells from maximum projection images.This function uses a band pass filter (only
    allowing medium sized, round shaped objects) consisting of two gaussian filters and a threshold caclulated from mean
    and empirical standard deviation. Detection of cells are calculated as the weighted centroid of segmented cells, if
    they are more then 75 pixel from the edge and are not much smaller the the average segmented cell.

    '''


    # segmentation of cells by broadband filter and threshold form stadart devaition and mean
    img = image.astype(float)
    img2 = gaussian(img, 15) / gaussian(img, 30)
    mu, std = np.mean(np.ravel(img2)), np.std(np.ravel(img2), ddof=1)
    if threshold == "otsu":
        mask = img2 > threshold_otsu(img2)
    if threshold == "mean_std":
        mask = img2 > mu + 5 * std

    labeled = measure_label(mask)
    regions = regionprops(labeled, intensity_image=img2) # calculates various properties of the segmented objects




    # filtering out objects that are unusually small and lie close to the image edge
    areas = []
    for r in regions:
        y, x = r.centroid
        if (75 < x < img.shape[1] - 75) and (75 < y < img.shape[0] - 75):
            areas.append(r.area)
    mu = np.mean(areas)  # mean and sd of all object areas not close to the image edges
    std = np.std(areas, ddof=1)
    detections = []
    coords = []
    for r in regions:
        y, x = r.centroid
        if (75 < x < img.shape[1] - 75) and (75 < y < img.shape[0] - 75) and (r.area > mu - 1.5*std): # filtering
            detections.append((x, y))  # stores x,y coordinates of object center (weighted by gauss-filtered image)
            coords.append(r.coords)  # stores slice of object positions
    detections = np.array(detections)

    return detections, labeled, coords






def detect_minProj_beads(image):

    '''
    parameters:
    image- numpy array, a minimum projection image
    returns:
    detections- array, object weighted centers, [object,xy_coordinates]
    coords- list, list of coordinates that cover the whole object area
    labeled_filter- labeled mask of segmentation (after filtering small objects)

    Detection and segemetation of beads from minimum projection images. Beads are detected in minimum projection images
    due to better visibility. Segemetnation is performed by  a broad band filter for small round objects followed by
    with otsu's method with an additional weight. Objects are rigorusly filter by size and distance to the image edge.

    '''
    # segmentation of beads
    img = image.astype(float)
    img2 = gaussian(img, 5)/gaussian(img, 6)  # broad band filter
    thresh=threshold_otsu(img2)*0.96 # otsu's threshold with aditional weight
    mask = img2 <   thresh
    labeled = measure_label(mask)
    regions = regionprops(labeled, intensity_image=img2)

    # filtering out all small objects. Only objects with an area greater then the mean plus 0.5 standart deviations are
    # allowed
    areas = []
    for r in regions:
        areas.append(r.area)
    mu = np.mean(areas)
    std = np.std(areas, ddof=1)
    labeled_filter=remove_small_objects(labeled,min_size=mu+0.5*std) # filtering for the size


    # filtering all detections close to the image edge
    regions = regionprops(labeled_filter, intensity_image=img2)
    detections = []
    coords=[]
    for r in regions:
        y, x = r.weighted_centroid
        if (75 < x < img.shape[1] - 75) and (75 < y < img.shape[0] - 75):
            detections.append((x, y) )# stores x,y coordinates of object center (weighted by gauss-filtered image)
            coords.append(r.coords)  # stores slice of object positions
    detections = np.array(detections)

    return detections , labeled_filter, coords



######bresenhams line algorithm from http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm
#also chek out https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
def get_line(start, end):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end
    start end as tupels of points
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points


def claculate_comp_matrix_prominence(values,distance):
    ## writing a comaprission matrix
    l = len(values)
    comp_matrix = np.empty((l, l))
    comp_matrix[
    :] = np.nan  ## squared matrix with shape l,l and third dimension (first entry is distance in pixel), second entry is prominenece
    for i in range(l):
        for j in range(l):
            if i > j:
                line = get_line(values[i], values[j])
                bg_distances = [distance[x, y] for x, y in line]  # getting distances to background on that line
                prominence = np.min(bg_distances) / np.min(
                    [bg_distances[0], bg_distances[-1]])  ### % of maximal indentation bewteen the maxima
                comp_matrix[i, j] = prominence
    return comp_matrix


def claculate_comp_matrix_distance(values):
    ## writing a comaprission matrix
    l = len(values)
    comp_matrix = np.empty((l, l))
    comp_matrix[
    :] = np.nan  ## squared matrix with shape l,l and third dimension (first entry is distance in pixel), second entry is prominenece
    for i in range(l):
        for j in range(l):
            if i > j:
                line = get_line(values[i], values[j])
                pixels_dist = len(line)
                comp_matrix[i, j] = pixels_dist
    return comp_matrix


def diff_img_med(img1, img2):


    img1_med = cv.medianBlur(img1, 21)
    img2_med = cv.medianBlur(img2, 21)
    img1_blurr = gaussian(np.array(img1_med, dtype="float64"),sigma=1)
    img2_blurr = gaussian(np.array(img2_med, dtype="float64"), sigma=1)

    diff_blurr = np.array(img1_blurr, dtype="float64") - np.array(img2_blurr, dtype="float64")  ## data types
    diff_blurr = norm(diff_blurr)
    return diff_blurr


def diff_img_sobel(img1, img2):
    grad_img1 = sobel(img1)
    grad_img_gauss1 = gaussian(grad_img1, sigma=10)
    grad_img2 = sobel(img2)
    grad_img_gauss2 = gaussian(grad_img2, sigma=10)
    diff = norm(grad_img_gauss1 - grad_img_gauss2)
    return diff


def norm(img):
    img_n=img-np.min(img)
    img_n=img_n/np.max(img_n)
    img_n*=255
    return img_n