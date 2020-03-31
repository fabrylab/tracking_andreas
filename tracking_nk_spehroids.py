from PIL import Image

from pyTrack.database_functions import setup_masks_and_layers, add_images
from pyTrack.detection_functions import cdb_add_detection, detect_diff, diff_img_sobel
from pyTrack.stitching_ants import stitch
from pyTrack.tracking_functions import tracking_opt
from pyTrack.utilities import *


def stitch_old(db, minutes_per_frames=5,type="track"):
    '''
    parameters:
    db- a cdb database obeject, containing Track_type markers
    returns:
    stiched_id- a list of all track ids, that have been stitched together

    Stitching tracks. Tracks are be joined together if marker in start and end frame have a small euclidean and temporal
    (difference of frames) distance. Tracks are joined by calculating a score matrix and finding the best matches. The score is
    composed of the euclidean distance added with the temporal distance, weighted by a factor. No tracks further then 10
    frames a part and only tracks with no temporal overlap will be stitched. Additionaly a maximal score  is set from
    the average velocity of all track. stitching will result in a new track with  nans at the marker position between
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

def add_diff_image(outputfolder,layer1="images",layer2="diff_images"):
    for i, frame in tqdm(enumerate(range(db.getImageCount() - 1))):
        # making and saving difference images
        img1 = db.getImage(layer="images", frame=frame)
        img2 = db.getImage(layer="images", frame=frame + 1)
        diff = diff_img_sobel(img1, img2)
        im = Image.fromarray(diff)
        name_img = 'diff' + str(i).zfill(4) + ".tif"
        im.save(os.path.join(outputfolder, name_img))
        db.setImage(filename=name_img, path=1, layer="diff_images", sort_index=frame)

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

def tracks_to_dict(t_type="trackpositive_detections"):
    tracks_dict=defaultdict(list)  ### improve by using direct sql query or by iterating through frames
    for i,t in enumerate(db.getTracks(t_type)):
         for m in t.markers:
              tracks_dict[i].append([m.x, m.y, m.image.sort_index])
    return tracks_dict

def write_tracks_dict_to_db(db, tracks_dict, marker_type):
    for t_id, values in tracks_dict.items():
        new_track = db.setTrack(marker_type)  # produces new track for marker_type
        xs = [x[0] for x in values]
        ys = [x[1] for x in values]
        frames = [x[2] for x in values]
        db.setMarkers(frame=frames, type=marker_type, x=xs, y=ys, track=new_track,
                      text="track_" + str(t_id))


folder=r'/home/user/Desktop/biophysDS/abauer/test_data_spheroid_spheroid_nk_migration/'
#images = glob.glob(r'/home/user/Desktop/biophysDS/dboehringer/Platte_4/4.3.19_NK_Lub11Sph_Tina/data/*')

outputfolder=r'/home/user/Desktop/biophysDS/abauer/test_data_spheroid_spheroid_nk_migration/'
file_list_dict=list_image_files(folder)

markers = {"positive_detections":"#00FF00","negative_detections":"#FF0000"}
masks =  {"positive_segmentation":["#0000FF",2] ,"negative_segmentation":["#00FF00",1],"overlapp":["#FF0000",3]}
layers =  ["images"]



##parameters:
max_tracking_dist = 100 # frames pixel previous value
min_frame_dist_stitching = 0 # frames no temporal overlapp
max_frame_dist_stitching = 5 # frames previous value??
max_dist_stitching = 80 # in pixels, not yet optimized
min_track_length = 4 # filtering small tracks /maybe 2 or 3 is also ok?



for name in file_list_dict.keys():
    print("analysing---" + name)
    print('--> Preprocessing')
    images = [os.path.join(folder, x) for x in file_list_dict[name]]  # full path to image
    # images = natsorted(images) # sorting the images, to get correct sorting for repetions

     # setting up the data base and adding images
    db_name = name + "database.cdb"
    db = setup_masks_and_layers(db_name, outputfolder, markers, masks, layers)
    add_images(db, images)



    # makeing diffrence images from frame i to frame i+1 #### could speed up dramatically by loading images from disc, not from clickpoints
    add_diff_image(outputfolder, layer1="images", layer2="diff_images")

    print('--> Detection')
    for i, frame in tqdm(enumerate(range(db.getImageCount() - 1))):
        cdb_add_detection(frame, db, detect_diff, cdb_types=["positive_detections","negative_detections"],
                          layer="diff_images", detect_type="diff")


    print('-->Tracking')
    tracking_opt(db, max_tracking_dist, type="positive_detections", color="#00FF00")
    tracking_opt(db, max_tracking_dist, type="negative_detections", color="#0000FF")

    print('-->Reading tracks from data base')
    tracks_dict1 = tracks_to_dict(t_type="trackpositive_detections")
    tracks_dict2 = tracks_to_dict(t_type="tracknegative_detections")

    # copy database before stitching
    copyfile(os.path.join(folder, db_name), os.path.join(folder, "not_stitched" + db_name))


    print('-->Stitiching')
    tracks_stitched1, stitched_id, gaps, old_ids = stitch(tracks_dict1, f_min=min_frame_dist_stitching
                                                          , f_max=max_frame_dist_stitching, s_max=max_dist_stitching, method="sparse")
    tracks_stitched2, stitched_id, gaps, old_ids = stitch(tracks_dict2, f_min=min_frame_dist_stitching
                                                          , f_max=max_frame_dist_stitching, s_max=max_dist_stitching, method="sparse")

     # filtering small tracks:
    tracks_stitched1 = {key:value for key,value in tracks_stitched1.items() if len(value) > min_track_length}
    tracks_stitched2 = {key: value for key, value in tracks_stitched2.items() if len(value) > min_track_length}

    #### removes all exisitng markers from the database!!!!!!!!!!
    db.deleteMarkers()
    db.deleteTracks()
    
    write_tracks_dict_to_db(db, tracks_stitched1, "trackpositive_detections")
    write_tracks_dict_to_db(db, tracks_stitched2, "tracknegative_detections")

    db.db.close()

