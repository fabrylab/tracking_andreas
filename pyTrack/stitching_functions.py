import numpy as np
from tqdm import tqdm
import clickpoints
import copy



def stitch(db, minutes_per_frames=5):
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

    tracks=db.getTracks()
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
    r_max=mean_displacement*10
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
    distance_matrix_time_select[(distance_matrix_time < -50) + (distance_matrix_time>=0)]=np.nan
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


