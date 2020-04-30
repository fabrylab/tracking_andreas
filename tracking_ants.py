# tracking of ants on a plate, now with creating a clickpoints data base.
from PIL import Image
from scipy.ndimage import zoom
from pyTrack.database_functions import *
from pyTrack.segmentation_functions import segementation_sd
from pyTrack.filters import filter_dog, fitler_objects_size
from pyTrack.detection_functions import cdb_add_detection, detect, simple_detection
from pyTrack.stitching_ants import stitch
from pyTrack.tracking_functions import tracking_opt
from pyTrack.utilities import *
import re
import pickle

#/home/user/Desktop/biophysDS/dboehringer/ants_andi/25_07_19_ants2/

# folders and image selction
folder=    r'/home/user/Desktop/biophysDS/dboehringer/ants_andi/25_07_19_ants2/'
outputfolder=     r'/home/user/Desktop/ants_29_04_2020'
im_selector="img(\d{6}).jpg" # regex to filter and sort images/ sortnig is controlled by choosing a group with "()", this group must
# contain ONLY integers

# mask for area that should be ignored for tracking:
mask_area=np.load("/home/user/Desktop/biophysDS/abauer/test_ants/mask.npy")

# sepcifcations for detection and segmentation
detect_kwargs = {"normalizing" : (0.1,99.9), # normalizing by removing quantiles #### (0.1,99.99) in original ants tracker
                 "filter": filter_dog, # filtering with difference of gaussian fucntions
                 "filter_kwargs" : {"s1": 2, "s2": 0.5}, # sigmas for dog filtering
                 "segmentation": segementation_sd, # segmentation with threshold based on standard deviation and mean of pixel intensities
                 "segmentation_kwargs": {"f":6, "mask_area": mask_area,"min_treshold":0.05}, # factor to standard deviation, and the specific area wheredetection is allowed
                  "mask_filter": fitler_objects_size, # second filtering step that removes objects according to their size
                 "mask_filter_kwargs":{"min_size": 8,"max_size":50}, # minimal size of objects  #### 5 in original ants tracker
                 "detection": simple_detection} # detection by weighted centroid
### I see some ants are detected twice/ but this is probabaly not a big deal

# tracking and stitching parameters
max_tracking_dist = 80 # original was 150
min_frame_dist_stitching = -2
max_frame_dist_stitching = 8
max_dist_stitching = 30 # very low number
min_track_length = 5


# database properties
markers = {"detections":"#00FF00"}
tracks = {"tracks":"#0000FF"}
masks =  {"ants":["#0000FF",1]}
layers =  ["images"]
new_folder = None





db_name = "database.cdb" # databesename
images = [x for x in os.listdir(folder) if re.search(im_selector, x)]
images = sorted(images, key=lambda x: int(re.search(im_selector,x).group(1))) # hoep that works always

images=images
# setting up the data base and adding images
createFolder(outputfolder)

#db = clickpoints.DataFile(os.path.join(outputfolder,db_name),"w")
#read_tracks_iteratively(db,track_types=["trackdetections"])



db = setup_masks_and_layers(db_name, folder, outputfolder, markers=markers, masks=masks, layers=layers, tracks=tracks)
add_images(db, images)

# detection
print('-->Detection')
for frame in tqdm(range(0, db.getImageCount() - 1)):
    cdb_add_detection(frame, db, detect, cdb_types="detections", detct_kwargs=detect_kwargs)

# tracking
print('-->Tracking')
tracks_dict = tracking_opt(db, max_tracking_dist, type="detections")
with open(os.path.join(outputfolder,"tracks_not_stiched.pickle"),"wb") as f:
    pickle.dump(tracks_dict,f, protocol=pickle.HIGHEST_PROTOCOL)
write_tracks_dict_to_db(db, tracks_dict, "tracks", method=1)
# copy database before stitching
n_db_name = os.path.join(outputfolder, "stitched" + db_name)
db.db.close() # not closing this database causes "meta" table to not be copied
copyfile(os.path.join(outputfolder, db_name), n_db_name)

print('-->Reading tracks from data base')

n_db = clickpoints.DataFile(n_db_name)
tracks_dict = read_tracks_iteratively(n_db, track_types=["tracks"], sort_id_mode=1, end_frame=None)
print('-->Stitiching')
tracks_stitched1, stitched_id, gaps, old_ids = stitch(tracks_dict, f_min=min_frame_dist_stitching
                                                      , f_max=max_frame_dist_stitching, s_max=max_dist_stitching,
                                                      method="sparse")
# filtering small tracks:
tracks_stitched1 = {key: value for key, value in tracks_stitched1.items() if len(value) > min_track_length}

with open(os.path.join(outputfolder,"tracks_stiched.pickle"),"wb") as f:
    pickle.dump(tracks_stitched1,f, protocol=pickle.HIGHEST_PROTOCOL)
#### removes all exisitng markers from the database!!!!!!!!!!
n_db.deleteMarkers()
n_db.deleteTracks()
print("writing stitched tracks")
write_tracks_dict_to_db(n_db, tracks_stitched1, "tracks", method=1)

n_db.db.close()