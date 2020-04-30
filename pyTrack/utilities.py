import clickpoints
import numpy as np
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from skimage.measure import label as measure_label
from skimage.morphology import label as morph_label
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.morphology import watershed
from skimage.measure import regionprops
import re
import os
from collections import defaultdict
import time
import copy
from shutil import copyfile
import sys
import datetime
from skimage.filters import threshold_otsu



def make_iterable(value):
    if not hasattr(value, '__iter__') or isinstance(value,str):
        return [value]
    else:
        return value

def split_path_with_os(folder):
    if not os.path.split(folder)[1]=="":
        parts = [os.path.split(folder)[1]]
        remaining = [os.path.split(folder)[0]]
    else:
        remaining1=os.path.split(folder)[0]
        parts = [os.path.split(remaining1)[1]]
        remaining = [os.path.split(remaining1)[0]]


    while True:
        path_part=os.path.split(remaining[-1])[1]
        if path_part == "":
            break
        parts.append(path_part)
        remaining.append(os.path.split(remaining[-1])[0])

    return parts

from functools import reduce
from operator import getitem

def make_iterable(value):
    if not hasattr(value, '__iter__') or isinstance(value,str):
        return [value]
    else:
        return value

def split_path_to_dict(folder, files, path_dict=None):
    parts = split_path_with_os(folder)
    reduce(getitem, parts[:-1], path_dict)[parts[-1]] = files # not entirely sure how this works..

def get_nested_defaultdict():
    # unlimited depth only
    tree = lambda: defaultdict(tree)
    return tree()

def make_iterable(value):
    if not hasattr(value, '__iter__') or isinstance(value, str):
        return [value]
    else:
        return value

    #else:
    #    return dict

def make_files_dict(folder):
    files_dict = {}
    for subdir, dirs, files in os.walk(folder):
        files_dict[subdir] = files
    return files_dict


def convert_to_defaultdict(old_dict,default=None):
    if callable(default):
        d = defaultdict(default)
    else:
        d = defaultdict(lambda: default)
    for key,value in old_dict.items(): # expend to arbirary number of keys...
        d[key]=value
    return d

path_filters = {"file": ["rep","Fluo",".tif"], 1:"", "full_path":"SphInv"}  # includes negative filters # e.g. "/^((?!exclude).)*$/"
sort_keys = {"file": [("_pos(\d+)",2)], 1:[("",None)], "full_path":[("(well\d+)",1),("(.*)",0)]} # always tupel with (key, level in dict), must enter list here!!!!!
neigbouring_paths = {}


def check_folders(folder,path_filters,sort_keys):
    folder_parts = split_path_with_os(folder)
    m_specific = []

    for folder_level,part_folder in enumerate(folder_parts): # starts with highest folder
        # all patterns need to be found in specific folder
        m_specific.append(all([re.search(pattern,part_folder) is not None for pattern in path_filters[folder_level]]))
        # also make sure sort keys are present
        m_specific.append(all([re.search(sk[0], part_folder) is not None for sk in sort_keys[folder_level]]))
    # patterns for full path must match somewhere in the complete folder tree
    m_general=all([re.search(pattern, folder) is not None for pattern in path_filters["full_path"]])
    m_general_sort_key=all([re.search(sk[0], folder) is not None for sk in sort_keys["full_path"]])

    return all(m_specific+[m_general]+[m_general_sort_key]) # both conditions must be true

def normalizing(img,lq=0,uq=100):
    img = img - np.percentile(img, lq)  # 1 Percentile
    img = img / np.percentile(img,uq)  # norm to 99 Percentile
    img[img < 0] = 0.0
    img[img > 1] = 1.0
    return img

'''
def check_file(f,path_filters):
    f=
    f2=


   return all([re.search(pattern, f) is not None for pattern in path_filters["file"]])
 m_general_sort_key=all([re.search(sk[0], folder) is not None for sk in sort_keys["full_path"]])

def identify_sort_keys(folder,file, sort_keys):
    # note: all files and folders must be guaranteed to have these patterns by steps bevor this step

    file_keys=[(re.search(sk[0], file).group(1),sk[1])  for sk in sort_keys["file"]]
    folder_keys=[(re.search(sk[0], folder).group(1),sk[1])  for sk in sort_keys["full_path"]]
    folder_specific_keys=[]
    folder_parts = split_path_with_os(folder)
    for folder_level, part_folder in enumerate(folder_parts):  # starts with highest folder
        if sort_keys[folder_level] is not None:
            folder_specific_keys.append([(re.search(sk[0], part_folder).group(1),sk[1]) for sk in sort_keys[folder_level]])




def collect_files(folder,selector_path="SphInv",selectors_file=["rep","Fluo"],negative_selectors_file=[]):

    path_filters=convert_to_defaultdict({key: make_iterable(value) for key,value in path_filters.items()},"")
    sort_keys = convert_to_defaultdict({key: make_iterable(value) for key, value in sort_keys.items()},[("",None)])
    neigbouring_paths = convert_to_defaultdict({key: make_iterable(value) for key, value in neigbouring_paths.items()},"")
    files_dict = make_files_dict(folder)

    for folder,files in files_dict.items():
        if check_folders(folder,path_filters,sort_keys):
            for f in files:
                if check_file(f,path_filters,sort_keys):
                    file1=f
                    folder1=folder
                    keys=

                    reduce(getitem, parts[:-1], path_dict)[parts[-1]] = files  # not entirely sure how this works..



        if "Analyzed_Data" in subdir:
            continue
        # checking if sphforce folder also exists

        bf_check = any(["SphForce" in x for x in dirs]) and any(["SphInv" in x for x in dirs])
        if bf_check:  # also checks if key already exists
            experiment = subdir
            print(experiment)

        ## stops iteration if not in correct folder tree
        if not bf_check and "SphForce" not in subdir and "SphInv" not in subdir:  #
            continue


        ## stops iteration if keyword is missing: diffrentiate between images from Sph Force and SphINv series
        if not all([x in subdir for x in selector_path]):
            continue

        # finds well identifier, doesnt search for files if well is not in the current subdirectory
        if not "well" in os.path.split(subdir)[1]:
            continue

        well = re.match("(well\d+)", os.path.split(subdir)[1])
        print(well)
        well_id = well.group(1)

        if well and not well_id in files_dict[experiment]:
            files_dict[experiment][well_id] = {}

        file_list = os.listdir(subdir)  # list all files (not directories) in any subdirectory
        if len(negative_selectors_file)>0:
            file_list_f = [x for x in file_list if x.endswith('.tif') and all(selector in x for selector in selectors_file) and not any(selector in x for selector in negative_selectors_file)]  # select all .tif images with rep and fluo in their name
        else:
            file_list_f = [x for x in file_list if
                           x.endswith('.tif') and all(selector in x for selector in selectors_file)]
        ## extract position number:

        search_list = [re.search(".*_pos(\d+).*", x) for x in file_list_f]

        # extracts the whole string of the filename except "rep" and all following numbers
        positions = [x.group(1) for x in search_list]
        positions = list(set(positions))  # list all unique strings

        try:  # sorting by position argument for better representation in output txt. file
            positions.sort(key=lambda x: re.search("(\d{1,3})", x).group(1))
        except:
            print("no position found in ", subdir)
        # creating a dictinary where keys are the future output filenames and values are all associated input filenames (parts of them) (for mean blending)
        for pos in positions:
            files_dict[experiment][well_id][pos.zfill(3)] = [os.path.join(subdir,file) for file in file_list_f if "pos"+pos in file]

    files_dict = remove_empty_keys(files_dict)

    return files_dict












'''
def ndargmin(array):
    return np.unravel_index(np.nanargmin(array), array.shape)



def createFolder(directory):
    '''
    function to create directories, if they dont already exist
    '''
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

    return directory



#function to convert none object from re.search to empty string

def get_group(s=None,group_number=1):

    '''
    parameters:
    s- a re.search object
    group_number- integer, specifies which group of the re.search should be returned
    returns:
    string of the group of the re.search object

    This function is used to return empty strings ("") froma re.search object if the object is empty.
    '''

    if s is None:
        return ''
    if s.group(group_number) is None:
        return ''
    else:
        return s.group(group_number)




