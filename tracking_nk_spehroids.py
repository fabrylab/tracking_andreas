from PIL import Image
from scipy.ndimage import zoom
from pyTrack.database_functions import setup_masks_and_layers, add_images
from pyTrack.detection_functions import cdb_add_detection, detect_diff, diff_img_sobel
from pyTrack.stitching_ants import stitch
from pyTrack.tracking_functions import tracking_opt
from pyTrack.utilities import *

def add_single_diff_image(frame ,outputfolder=None, layer1="images", layer2="diff_images",save=True,  save_type=".tif",save_diff_quality=None):
    img1 = db.getImage(layer=layer1, frame=frame)
    img2 = db.getImage(layer=layer1, frame=frame + 1)
    diff = diff_img_sobel(img1, img2)

    if save:
        name_img = 'diff' + str(frame).zfill(4) + save_type
        if save_type in[".jpeg",".png"]: # converting to appropriate datatype
            diff_save = (normalizing(diff)*255).astype("uint8")
        else:
            diff_save = diff
        im = Image.fromarray(diff_save)
        # saving with jpeg compression
        if isinstance(save_diff_quality,(float,int)) and save_type==".jpeg":
            im.save(os.path.join(outputfolder, name_img,), format='JPEG',quality=save_diff_quality)
        else:
            im.save(os.path.join(outputfolder, name_img))
        db.setImage(filename=name_img, path=2, layer=layer2, sort_index=frame)
    return  diff

def add_diff_image(outputfolder, add_name="", layer1="images", layer2="diff_images"):
    new_folder = createFolder(os.path.join(outputfolder, "diff" + add_name))
    db.setPath(new_folder,2)
    for i, frame in tqdm(enumerate(range(db.getImageCount() - 1))):
        # making and saving difference images
        img1 = db.getImage(layer= layer1, frame=frame)
        img2 = db.getImage(layer= layer1, frame=frame + 1)
        diff = diff_img_sobel(img1, img2)
        im = Image.fromarray(diff)
        name_img = 'diff' + str(i).zfill(4) + ".tif"
        im.save(os.path.join(new_folder, name_img))
        db.setImage(filename=name_img, path=2, layer=layer2, sort_index=frame)

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

outputfolder=r'/home/user/Desktop/biophysDS/abauer/test_data_spheroid_spheroid_nk_migration2/'
file_list_dict=list_image_files(folder)

markers = {"positive_detections":"#00FF00","negative_detections":"#FF0000"}
masks =  {"positive_segmentation":["#0000FF",2] ,"negative_segmentation":["#00FF00",1],"overlapp":["#FF0000",3]}
layers =  ["images"]
new_folder = None


##parameters:
max_tracking_dist = 100 # frames pixel previous value
min_frame_dist_stitching = 0 # frames no temporal overlapp
max_frame_dist_stitching = 5 # frames previous value??
max_dist_stitching = 80 # in pixels, not yet optimized
min_track_length = 4 # filtering small tracks /maybe 2 or 3 is also ok?
save_diffs = True # sets whether to save the diff images or not
save_diff_quality = 40 # quality when saving as jpeg range from 1 to 75 is advised
save_diff_type=".jpeg" # can further help with compression

for name in file_list_dict.keys():
    print("analysing---" + name)
    print('--> Preprocessing')
    images = [os.path.join(folder, x) for x in file_list_dict[name]]  # full path to image
    # images = natsorted(images) # sorting the images, to get correct sorting for repetions

     # setting up the data base and adding images
    db_name = name + "db.cdb"
    db = setup_masks_and_layers(db_name, folder, outputfolder, markers, masks, layers)
    add_images(db, images)



    # makeing diffrence images from frame i to frame i+1 #### could speed up dramatically by loading images from disc, not from clickpoints
    #add_diff_image(outputfolder, add_name=name, layer1="images", layer2="diff_images")
    if save_diffs:
        new_folder = createFolder(os.path.join(outputfolder, "diff" + name))
        db.setPath(new_folder, 2)

    print('--> Detection')
    for frame in tqdm(range(db.getImageCount() - 1)):
        image = add_single_diff_image(frame, new_folder, layer1="images", layer2="diff_images",save=save_diffs
                                      ,save_diff_quality=save_diff_quality,
                                       save_type=save_diff_type)
        cdb_add_detection(frame, db, detect_diff, cdb_types=["positive_detections","negative_detections"],
                          layer="diff_images", detect_type="diff",image=image)


    print('-->Tracking')
    tracking_opt(db, max_tracking_dist, type="positive_detections", color="#00FF00")
    tracking_opt(db, max_tracking_dist, type="negative_detections", color="#0000FF")

    print('-->Reading tracks from data base')
    tracks_dict1 = tracks_to_dict(t_type="trackpositive_detections")
    tracks_dict2 = tracks_to_dict(t_type="tracknegative_detections")

    # copy database before stitching
    copyfile(os.path.join(outputfolder, db_name), os.path.join(outputfolder, "not_stitched" + db_name))


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

