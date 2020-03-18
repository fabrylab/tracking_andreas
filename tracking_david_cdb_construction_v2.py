import sys
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from skimage.measure import label as measure_label
from skimage.morphology import remove_small_objects
from skimage.filters import gaussian,sobel,laplace, threshold_otsu,threshold_local,threshold_niblack
import os
from skimage.measure import regionprops
import re
from shutil import copyfile
import cv2 as cv
import clickpoints
from PIL import Image

from pyTrack.detection_functions import cdb_add_detection, detect_diff, diff_img_sobel
from pyTrack.utilities import *
from pyTrack.database_functions import setup_masks_and_layers

def tracking_opt(db, max_dist,type="",color="#0000FF"):

    all_markers = {}  # write this at segemntation and detection

    # dictionary [frame][marker id][(x,y) position)
    for frame in range(db.getImageCount()):
        all_markers[frame] = []
        for marker in db.getMarkers(frame=frame, type=type):
            all_markers[frame].append([marker.x, marker.y])

    # dictionary with [track id][(marker id, frame)]

    tracks = {id: [(0, np.array(position))] for (id, position) in
              enumerate(all_markers[0])}  # also initializing first values
    ids = np.array(list(tracks.keys()))  # list of track ids, to associate with a merker

    for frame in range(1, db.getImageCount() - 1):
        markers1_pos = np.array(all_markers[frame - 1])
        markers2_pos = np.array(all_markers[frame])
        if len(markers2_pos) == 0:  # if sttatement if no detections are found in the next frame
            continue
        if len(markers1_pos) == 0:  # if sttatement if no detections are found in the previous frame
            remaining = np.arange(len(markers2_pos))  # remaining markers from markers2_pos
            # new track entries
            for i, ind in enumerate(remaining):
                tracks[max_id + i] = [(frame, markers2_pos[ind])]
                ids_n[remaining] = max_id + i
            ids = ids_n  # overwriting old ids assignement
            continue

        distances = np.linalg.norm(markers2_pos[None, :] - markers1_pos[:, None],
                                   axis=2)  # first substracting full all values in matrix1 with all values n matrix2,
        # then norm ofer appropriate axis # distances has
        # matrix -->markers2 (axis2)
        # |
        # |
        # makers1 (axis1)
        min_value = 0
        row_ind, col_ind = [], []
        while np.nanmin(distances) < max_dist:
            min_pos = list(np.unravel_index(np.nanargmin(distances), distances.shape))
            distances[min_pos[0], :] = np.nan
            distances[:, min_pos[1]] = np.nan
            row_ind.append(int(min_pos[0]))
            col_ind.append(int(min_pos[1]))

        row_ind, col_ind = np.array(row_ind), np.array(
            col_ind)  # finds optimal adssigment , check if this is faster then nan method

        ids_n = np.zeros(len(markers2_pos))  # list of track ids the markers from markers2_pos have been assigned to
        for id, ind in zip(ids[row_ind], col_ind):
            tracks[id].append((frame, markers2_pos[ind]))
            ids_n[ind] = id

        remaining = np.arange(len(markers2_pos))  # remaining markers from markers2_pos
        remaining = remaining[~np.isin(remaining, col_ind)]
        max_id = np.max(list(tracks.keys()))

        # new track entries
        for i, ind in enumerate(remaining):
            tracks[max_id + i + 1] = [(frame, markers2_pos[ind])]
            ids_n[ind] = max_id + i + 1
        ids = np.array([int(x) for x in ids_n])  # overwriting old ids assignement

    # settig new tracks



    db.setMarkerType(name="track"+type, color=color, mode=db.TYPE_Track)
    for id, values in tracks.items():
        print(id, values)
        new_track = db.setTrack('track'+type)
        xs = [x[1][0] for x in values]
        ys = [x[1][1] for x in values]
        frames = [x[0] for x in values]
        db.setMarkers(frame=frames, type='track'+type, x=xs, y=ys, track=new_track,
                      text="track_" + str(id))


def stitch(db, minutes_per_frames=5,type="track"):
    '''
    parameters:
    db- a cdb database obeject, containing Track_type markers
    returns:
    stiched_id- a list of all track ids, that have been stitched together

    Stitching tracks. Tracks are be joined together if marker in start and end frame have a small euclidean and temporal
    (difference of frames) distance. Tracks are joined by calculating a score matrix and finding the best matches. The score is
    composed of the euclidean distance added with the temporal distance, weighted by a factor. No tracks further then 10
    frames a part and only tracks with no temporal overlap will be stitched. Additionaly a maximal score  is set from
    the average velocity of all tracks.titching will result in a new track with  nans at the marker position between
    the two old tracks.
    '''

    tracks=db.getTracks(type=type)
    length_tracks=len(tracks)

    # creates a dictionary with keys-track ids and values: all marker coordinates in the track, start frame and endframe
    track_dict = {}
    for track in tracks:
        track_dict[track.id] = (track.points[:], track.frames[0], track.frames[-1])

    # calculating the mean velocity of all tracks
    mean_displacement=0
    for id in track_dict:
        vectors=track_dict[id][0][:-1]-track_dict[id][0][1:]
        if len(vectors)>0:
            norm=np.mean(np.linalg.norm(vectors,axis=1))
            mean_displacement +=norm
    mean_displacement=mean_displacement/length_tracks

    # sets the maximal for stitching to be allowed.
    r_max=mean_displacement*5
    # sets a weighting for the temporal distance of two tracks
    time_punishment = 3/5

    # setting up distance matrices
    distance_matrix_space=np.zeros((length_tracks,length_tracks))
    distance_matrix_space.fill(np.nan)
    distance_matrix_time=np.zeros((length_tracks,length_tracks))
    distance_matrix_time.fill(np.nan)
#matrix shape
    # --->  start-end distance
    #|
    #end-start distance (negative)


    # calculating the temporal and euclidean distaces for all tracks
    for i,key1 in enumerate(track_dict.keys()):
        for j,key2 in enumerate(track_dict.keys()):
            if key1 != key2: # avoid calculation of start end distance within one track
                time_dist=(track_dict[key1][2]-track_dict[key2][1])*minutes_per_frames   # end frame of id key - start frame of id key2
                # euclidean distance of end track,id key and start track,id key2
                space_dist=np.sqrt(np.sum((track_dict[key1][0][-1]-track_dict[key2][0][0])**2))
                distance_matrix_space[i,j]=space_dist
                distance_matrix_time[i,j]=time_dist



    # excluding track pairs with temporal distance >10 and temporal overlapp:
    distance_matrix_time_select=copy.deepcopy(distance_matrix_time)
    distance_matrix_time_select[(distance_matrix_time < -25) + (distance_matrix_time>=0)]=np.nan
    # calculates the final score for stitching. Note that time distance has exclusively negative values here.
    stitch_score_matrix=distance_matrix_space - distance_matrix_time_select*time_punishment


    # finding track pairs that need to be stiched. The score matrix is is iteratively searched for the best
    # matches, until the maximal allowed score r_max is reached
    stitched=[]
    stitched_id=[]
    while True:
        # finding  indices of minimum
        minimum_pos = np.unravel_index(np.nanargmin(stitch_score_matrix,axis=None), stitch_score_matrix.shape)
        minimum = stitch_score_matrix[minimum_pos]
        if minimum > r_max:
            break
        stitch_score_matrix[minimum_pos[0],:].fill(np.nan)  # deleting stitched end and starts entries in the score matrix
        stitch_score_matrix[:,minimum_pos[1]].fill(np.nan)
        id1=list(track_dict.keys())[minimum_pos[0]] # writing stiched ids inti a list
        id2=list(track_dict.keys())[minimum_pos[1]]
        stitched.append(list(minimum_pos))
        stitched_id.append((id1,id2))  # stitch stiched[0] (start) to stiched[1] end


    # merging all tracks found to be stiched. Merging will leave nans between the two merged tracks.
    stitched=np.array(stitched)
    print("merging ", len(stitched), " tracks")
    for i in tqdm(range(len(stitched))):
        id1,id2=stitched[i]
        # updating the annotation of tracks and "manual merging"
        merged_markers=tracks[id2].markers
        new_annotation="track_"+str(tracks[id1].id)
        for marker in merged_markers:
            db.deleteMarkers(id=marker.id)
            db.setMarker(x=marker.x, y=marker.y, text=new_annotation,
                         frame=marker.image.id - 1, type=tracks[id1].type, track=tracks[id1])

       # tracks[id1].merge(tracks[id2])     ### changing track annotation tracks
        # merging will delete the track of id2 and append to id1. Therefore stitched needs to be updated
        stitched[stitched==id2]=id1

    return stitched_id



def list_image_files(directory):

    '''
        returns:
        file_list_dict_max- a dictionary with keys:possible cdb filenames, values: all filenames of maximum images
        this function will:
        1) create a list of all Maximum projection images, matching with, file list filter.
        2) try to find a suitable name for the output cdb file.

    '''

    # searching for a suitable name for the cdb outputfile. Mostly anything except for the repetition is used.
    pos_match=re.compile('(.*_)rep\d*(_pos\d*)(_\d+){0,1}_')
    file_filter_max=re.compile('.*rep\d{0,4}.*\.tif') # filtes to get the correct max images
    target_files = os.listdir(directory) # reading all files from the directory
    target_files_filter_max = [x for x in target_files if file_filter_max.match(x)] # filtering for all max images
    # retrieving a possibel cdb filename. This could also support multiple experiments in one folder.
    identifiers_max = [(get_group(pos_match.search(x), 1), get_group(pos_match.search(x), 2),
         get_group(pos_match.search(x), 3)) for x in target_files_filter_max]
    identifiers_max=list(set(identifiers_max)) # getting unique values
    file_list_dict_max = {} # writing files to a dictionary, with: key=cdb_filename, values=image filenames
    for identifier in identifiers_max:
        cdb_name = identifier[0] + identifier[1] + identifier[2]
        # image files are sorted. This is necessary if the repetition number is not padded with zeros
        file_list_dict_max[cdb_name] = natsorted([x for x in target_files_filter_max if
                                    re.match(identifier[0] + 'rep\d*' + identifier[1] + str(identifier[2]) + '_.*', x)])
    return(file_list_dict_max)




folder=r'/home/user/Desktop/biophysDS/dboehringer/Platte_4/4.3.19_NK_Lub11Sph_Tina/data/'
import glob as glob
#images = glob.glob(r'/home/user/Desktop/biophysDS/dboehringer/Platte_4/4.3.19_NK_Lub11Sph_Tina/data/*')

outputfolder=r'/media/user/GINA1-BK/davids_stuff_12_02_2020/'
file_list_dict=list_image_files(folder)

markers = {"positive_detections":"#00FF00","negative_detections":"#FF0000","pre_track_detection":"#0000FF"}
masks =  {"positive_segmentation":["#0000FF",2] ,"negative_segmentation":["#00FF00",1],"overlapp":["#FF0000",3]}
layers =  ["images"]


for name in file_list_dict.keys():
        print("analysing---"+name)
        print('--> Preprocessing')
        images = [os.path.join(folder,x) for x in file_list_dict[name]] # full path to image
        #images = natsorted(images) # sorting the images, to get correct sorting for repetions

        db_name = name + "database.cdb"
        db=setup_masks_and_layers(db_name, outputfolder, markers, masks, layers)


        # writes images with sort index (cooresponding to frame) and correct path to cb file
        print('--> Preprocessing')
        for i, file in tqdm(enumerate(images),total=len(images)):
            image = db.setImage(filename=os.path.split(file)[1], path=1,layer="images",sort_index=i)

        print('--> Detection')
        for frame in tqdm(db.getImageCount()-1):
            # making and saving difference images
            img1 = db.getImage(layer="images", frame=frame)
            img2 = db.getImage(layer="images", frame=frame+1)
            diff = diff_img_sobel(img1, img2)
            im = Image.fromarray(diff)
            name_img ='diff' + str(i).zfill(4) + ".tif"
            im.save(os.path.join(outputfolder,name_img))
            image = db.setImage(filename=name_img, path=1, layer="diff_images", sort_index=frame)

            cdb_add_detection(frame, db, detect_diff, layer="diff_images", detect_type = "diff", image=diff)

        print('-->Tracking')
        r = 100
        tracking_opt(db, r,type="positive_detections",color="#00FF00")
        tracking_opt(db, r, type="negative_detections",color="#0000FF")


        print('-->Stitiching')


        copyfile(outputfolder+'/'+name+'bright_part.cdb',outputfolder+'/'+name+'bright_part_prestitch.cdb')

        stitched_id_p = stitch(db,type="trackpositive_detections")
        stitched_id_n = stitch(db, type="tracknegative_detections")

        db.db.close()


