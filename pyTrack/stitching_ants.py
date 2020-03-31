#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 21:07:35 2019

@author: andy
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import copy
from scipy.spatial import cKDTree
#from vizualisation_and_analysis import *
import time


def write_tracks2(tracks_arr,file):
    '''
    writes tacks to same text file format as "write_tracks", but does so all at once
    '''
    
    tracks_arr_w=copy.deepcopy(tracks_arr)
    tracks_arr_w[np.isnan(tracks_arr)]=-1 # replacing nans according to convention
    tracks_arr_w=tracks_arr_w.astype(np.unicode_) # converting to string
    
    with open(file,"w+") as f:
        for frame,l in enumerate(tracks_arr_w): # iterating through all rows
            line=[",".join(pos) for pos in l]
            line=str(frame)+"\t"+"\t".join(line)+"\n"
            f.write(line)
    
def write_times(times,file):
      with open(file,"w+") as f:
          for frame,t in times.items():
              f.write(str(frame)+"\t"+str(t)+"\n")
def vizualize_stitch_assignemnet(ids):
    m=np.zeros((np.max(ids)+1,np.max(ids)+1))
    m[ids[:,0],ids[:,1]]=1
    plt.figure()
    plt.imshow(m)

def fill_gaps(tracks,gaps):
    '''
    fills the tracks dictionary with position in gaps bridge while stitching
    '''
    n_tracks=copy.deepcopy(tracks)
    for t_id in tracks.keys():
        if len(gaps[t_id])>2:
            n_tracks[t_id].extend(gaps[t_id]) # appending gaps
            n_tracks[t_id].sort(key=lambda x: x[-1]) # sorting elements by frame
            
    return n_tracks


def  stitch_order(stitched_id):
    
    '''
    function to assemble the correct order in which to stitch tracks.
    '''
    
    if len(stitched_id)<1:
        return []
        
    # all tracks of wich the end is stitched, but not the start
    start_points=set(stitched_id[:,0])-set(stitched_id[:,1]) 
    # all tracks where only the start is stitched
    end_points=set(stitched_id[:,1])-set(stitched_id[:,0]) 
    stitching_lists=defaultdict(list)
    
    # going through the stitched list form start point, until endpoint is reached
    for sp in start_points:
        new_id=int(stitched_id[:,1][np.where(stitched_id[:,0]==sp)])
        stitching_lists[sp].append(new_id)
        while new_id not in end_points:
            new_id=int(stitched_id[:,1][np.where(stitched_id[:,0]==new_id)])
            stitching_lists[sp].append(new_id)
    return stitching_lists
            
def predict_points(pos1,pos2):
    '''
    interpolation of points between two points from stitching
    '''
    steps=pos2[-1]-pos1[-1] #pnumber oints to interpolate betwen pos 1 and pos2 
    dif_vec=np.array(pos2[:-1])-np.array(pos1[:-1])
    pos_new=[]
    for i in range(1,steps):
        pos_new.append(list(np.array(pos1[:-1])+dif_vec*i/(steps))+[pos1[-1]+i])
       
   # positions=[pos1] +pos_new +[pos2]
    return pos_new

def get_stitching_ids_array(tracks_dict,f_min=-2,f_max=10,s_max=300):

    n_tracks = len(tracks_dict.keys())
    # creating a dictionary with keys-track ids and values: all marker coordinates in the track, start frame and endframe
    stitch_dict = {}
    for track_id, positions in tracks_dict.items():
        positions = np.array(positions)
        stitch_dict[track_id] = (positions[:, np.array([0, 1])], positions[0, 2], positions[-1, 2])
    # sets the maximal for stitching to be allowed.
    # sets a weighting for the temporal distance of two tracks
    # setting up distance matrices
    distance_matrix_space = np.zeros((n_tracks, n_tracks)) + np.nan
    distance_matrix_time = np.zeros((n_tracks, n_tracks)) + np.nan
    # matrix shape
    # --->  start-end distance
    # |
    # end-start distance (negative)
    # calculating the temporal and euclidean distances for all tracks
    for i, key1 in tqdm(enumerate(stitch_dict.keys()), total=len(stitch_dict.keys())):
        for j, key2 in enumerate(stitch_dict.keys()):
            if key1 != key2:  # avoid calculation of start end distance within one track
                time_dist = -(stitch_dict[key1][2] - stitch_dict[key2][1])  # end frame of id key - start frame of id key2
                # euclidean distance of end track,id key and start track,id key2
                space_dist = np.sqrt(np.sum((stitch_dict[key1][0][-1] - stitch_dict[key2][0][0]) ** 2))
                distance_matrix_space[i, j] = space_dist
                distance_matrix_time[i, j] = time_dist
    # excluding track pairs with temporal distance >10 and temporal overlapp:
    distance_matrix_time_select = copy.deepcopy(distance_matrix_time)
    tf = (distance_matrix_time > f_max) + (distance_matrix_time < f_min)
    distance_matrix_time_select[tf] = np.nan
    distance_matrix_space[tf] = np.nan
    # calculates the final score for stitching. Note that time distance has exclusively negative values here.
    # slight addition for negative frame diffrences, to get correct order when two stitching directions are psoiible

    stitch_score_matrix = distance_matrix_space + ((distance_matrix_time_select < 0)) * 0.1

    # finding track pairs that need to be stiched. The score matrix is is iteratively searched for the best
    # matches, until the maximal allowed score r_max is reached
    stitched_id = []
    all_ids = list(stitch_dict.keys())
    while np.nanmin(stitch_score_matrix) < s_max:
        minimum_pos = np.unravel_index(np.nanargmin(stitch_score_matrix, axis=None), stitch_score_matrix.shape)
        id1 = all_ids[minimum_pos[0]]  # writing stitched ids into a list
        id2 = all_ids[minimum_pos[1]]
        # checking if the same connection has been made in reverse
        if [id2, id1] in stitched_id:
            # deleting stitched end and starts entry only for this pair in the score matrix
            stitch_score_matrix[minimum_pos[0], minimum_pos[1]] = np.nan
        else:
            # deleting stitched end and starts entries for all pairs
            stitch_score_matrix[minimum_pos[0], :].fill(np.nan)
            stitch_score_matrix[:, minimum_pos[1]].fill(np.nan)
            stitched_id.append([id1, id2])  # stitch stiched[0] (end) to stiched[1] start
    stitched_id = np.array(stitched_id)
    return stitched_id




def get_stitching_ids_sparse(tracks_dict,f_min=-2,f_max=10,s_max=300):


    # creating a dictionary with keys-track ids and values: all marker coordinates in the track, start frame and endframe
    stitch_dict = {}
    for track_id, positions in tracks_dict.items():
        positions = np.array(positions)
        stitch_dict[track_id] = (positions[:, np.array([0, 1])], positions[0, 2], positions[-1, 2])
    # cKD trees search to find suitable stitching candidates
    f_boundary = np.max(np.abs([f_max, f_min]))
    time_points_end = np.array([t[2] for t in stitch_dict.values()]) / f_boundary  #
    # also nomralizing this dimension, so that maximal
    # allowed distances is 1 (but no distinction ofr positivee and negative direction??)
    time_points_start = np.array([t[1] for t in stitch_dict.values()]) / f_boundary
    space_points_end = np.array([t[0][-1] for t in stitch_dict.values()]) / s_max
    space_points_start = np.array([t[0][0] for t in stitch_dict.values()]) / s_max
    cKD_start = cKDTree(np.vstack([space_points_start.T, time_points_start]).T)
    cKD_end = cKDTree(np.vstack([space_points_end.T, time_points_end]).T)
    neigbours = cKD_end.sparse_distance_matrix(cKD_start, max_distance=1,p=np.inf)## p gives "minkovski p-nomr, p=2 would be euclidean norm
    # p =np.inf give inifinty norm: distance of the closest dimension
    neigbours = neigbours.tocoo()  # conversion to coordiante matrix, for easy row and column extraction
    rows = neigbours.row  # ids of tracks that ends could be stitched
    cols = neigbours.col  # ids of tracks that starts could be stitched

    #### additinoal filters: necessary because we want to treat space and time dimensions diffrently and also
    # negative and positivve time dimensions diffrently

    # excluding entries with start and end in the same track, and exclusion for
    # second time condition
    data_time = time_points_start[cols] * f_boundary - time_points_end[rows] * f_boundary  # time diffrence in frames
    data_space = np.linalg.norm(space_points_start[cols] * s_max - space_points_end[rows] * s_max,
                                axis=1)  # eucledean distances

    f_time = np.logical_or(data_time > f_max, data_time < f_min)  # using time condition f_min and f_max
    f_same = (rows - cols) == 0  # excluding entries with start and end in the same track
    f_space =  data_space > s_max
    filter = ~(f_same + f_time + f_space) # logiacl or for all filters
    rows = rows[filter]  # filtering
    cols = cols[filter]
    data_time = data_time[filter]
    data_space = data_space [filter]
    # (optional) calculating a seperate score for sttiching

    ## adding a slight time punishment to adress reverse and forward stitching
    # ---> favors positive time direction
    time_add = (data_time < 0) * 0.1  # adds 0.1 pixel if time diffrence is smaller then zero
    stitch_scores = data_space + time_add  # score for sttiching, here just the euclidean distance


    ## stitching tracks according to stitch scores
    

    stitch_scores2 = copy.deepcopy(stitch_scores)
    rows2 = copy.deepcopy(rows)
    cols2 = copy.deepcopy(cols)
    stitched_id = []  # lsit of ids to be stitched
    init_length = len(cols2)
    while len(cols2) > 0:
        m = np.argmin(stitch_scores2)  # best match for stitching
        tid_start, tid_end = rows2[m], cols2[m]  # ids of tracks
        # nice little progress bar
        print("".join(["-"] * int(100 * (len(cols2) / init_length))) + " remaining track pairs: " + str(len(cols2)),
              end='\r')
        # check if the same pair was already stitched with reverse orientation
        if [tid_end, tid_start] in stitched_id:
            f = ~np.logical_and(rows2 == tid_start, cols2 == tid_end)  # filter only this pair
        else:
            stitched_id.append([tid_start, tid_end])  # append ids to stitching list
            f = ~np.logical_or(rows2 == tid_start,
                               cols2 == tid_end)  # remove all other matches for relevent start and ends
        rows2, cols2, stitch_scores2 = rows2[f], cols2[f], stitch_scores2[f]
    stitched_id=np.array(stitched_id)

    return stitched_id

def stitch(tracks_dict,f_min=-2, f_max=10, s_max=300, method="sparse"):
    '''

    stitching is intendend to connect interrupted tracks (no detection in some 
    frames in between). Here tracks ends and beginnings are only allowed to be 
    f_max frames appart, and may not overlapp. Then a score is calculated as
    eucleadean_distance - assumed_traveled_distances *time_punishment (set to 3/5)
    where the assumed_traveled_distances= frame_distance*speed

     param: tracks_dict: key:id, values [[x_pos,y_pos,frame],[],...], frame should be integer
    param: seconds_per_frame: length of a frame in seconds
    param: f_max: maximum number of frames a track end a beginning are allowed to be appart
    param: speed: assumed speed of moving bugs in pixels per frame
    param: s_max: maximal stitch score allowed to stich two tracks together
    '''

    if method=="sparse":
        stitched_id = get_stitching_ids_sparse(tracks_dict,f_min=f_min,f_max=f_max,s_max=s_max)
    else:
        stitched_id = get_stitching_ids_array(tracks_dict,f_min=f_min,f_max=f_max,s_max=s_max)

    stitching_lists = stitch_order(stitched_id)

    # merging all tracks found to be stiched. Merging will leave nans between the two merged tracks.
    
    tracks_stitched = defaultdict(list)# updated dictionary with stiched tracks
    gaps = defaultdict(list) # dictionary with the gaps
    
    if len(stitched_id)==0: # retunr unschaned and empty results if nothing is stitched
        return tracks_dict,stitched_id,gaps
     #
    # copying not stitched tracks
    not_stitched=set(list(tracks_dict.keys()))-set(stitched_id[:,0]).union(set(stitched_id[:,1]))
    for id in not_stitched:
        tracks_stitched[id]=copy.deepcopy(tracks_dict[id])
        gaps[id]=[] # nothing filled up in none stitched tracks
       
    for id_start,ids in stitching_lists.items():
        tracks_stitched[id_start]=copy.deepcopy(tracks_dict[id_start])
        for id in ids: # adding points fomr all tracks to be stitched
            tracks_stitched[id_start]+=tracks_dict[id] 
             
        # noting the gaps and pridicting points in the gaps
        gaps[id_start].extend(predict_points(tracks_dict[id_start][-1],tracks_dict[ids[0]][0]))
        for i in range(len(ids)-1): 
            gaps[id_start].extend(predict_points(tracks_dict[ids[i]][-1],tracks_dict[ids[i+1]][0]))
        #gaps[id_start]
        
    # give new_ids for tracks:
    old_ids=list(tracks_stitched.keys())
    tracks_stitched={i:values for i,values in zip(range(len(tracks_stitched.keys())),tracks_stitched.values())}

    # replacing overlapping points
    for key,values in tracks_stitched.items():
        values_s=sorted(values,key=lambda x: x[-1])# sorting the values according to their frame
        points=np.array(values_s)
        id_rm=np.where((points[1:,2]-points[:-1,2])==0)[0] # first overlapping entry
        new_points=[np.mean(np.array([points[i],points[i+1]]),axis=0) for i in id_rm]#
        for j,i in enumerate(id_rm):
            points[i]=new_points[j] # replacing with mean
        points=np.delete(points,np.array(id_rm)+1,axis=0) # removing row just 
        # after the replaced row
        tracks_stitched[key] = points# writing to new dictionary
    
    return tracks_stitched,stitched_id,gaps,old_ids

def remove_parralle_tracks(tracks_dict,tracks_arr,end_dist=30,mean_dist=30):
    
    tracks_dict_filtered=copy.deepcopy(tracks_dict)
    n_tracks=len(tracks_dict_filtered.keys())
    dists=np.zeros((n_tracks,n_tracks,3))+np.nan
    ids=np.array(list(tracks_dict_filtered.keys())) # note: after stitching track_id and row 
    #index in array are no longer the same
    
    # this comparison is directional: row is the index of the "smaller" track.
    
    for i,tid1 in enumerate(tracks_dict_filtered.keys()):
        # i is the coorect row in the tracks array, tid is the correct index in 
        # the tracks dictionary
        for j,tid2 in enumerate(tracks_dict_filtered.keys()):
            if i!=j: # dont calculate "distance with itself
                first_frame = int(tracks_dict_filtered[tid1][0][-1])
                last_frame =  int(tracks_dict_filtered[tid1][-1][-1])
                ps1=tracks_arr[first_frame:last_frame+1,i,:] # all points in this range
                ps2=tracks_arr[first_frame:last_frame+1,j,:] 
    
                dists[i,j,0]=np.linalg.norm(ps1[0]-ps2[0]) # distance in start points
                # returns nan if track ends/starts doesnt appear in the same frames
                dists[i,j,1]=np.linalg.norm(ps1[-1]-ps2[-1]) # distance in endpoints
                dists[i,j,2]=np.nanmean(np.linalg.norm(ps1-ps2,axis=1))
        
            
    # removal conditions:
    # logical and on all conditions    
    exclude=(dists[:,:,0]<end_dist)*(dists[:,:,1]<end_dist)*(dists[:,:,2]<mean_dist)
    # tracks to be excluded:
    
    # all ws  where any exclude matrix entr
    # is true corespond to track ids that must be excluded
    ex_ids=ids[np.where(exclude)[0]]
    for i in ex_ids:
        del(tracks_dict_filtered[i]) # deletes key from dictionary
        
        ##### bbuild on that...
    return tracks_dict_filtered
        
