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



# folders and image selction
folder=    r'/home/user/Desktop/biophysDS/abauer/test_ants/'
outputfolder=     r'/home/user/Desktop/biophysDS/abauer/test_ants/out'
im_selector="img(\d{6}).jpg" # regex to filter and sort images/ sortnig is controlled by choosing a group with "()", this group must
# contain ONLY integers

# mask for area that should be ignored for tracking:
mask_area=np.load("/home/user/Desktop/biophysDS/abauer/test_ants/mask.npy")

# sepcifcations for detection and segmentation
detect_kwargs = {"normalizing" : (0.1,99.9), # normalizing by removing quantiles #### (0.1,99.99) in original ants tracker
                 "filter": filter_dog, # filtering with difference of gaussian fucntions
                 "filter_kwargs" : {"s1": 2, "s2": 0.5}, # sigmas for dog filtering
                 "segmentation": segementation_sd, # segmentation with threshold based on standard deviation and mean of pixel intensities
                 "segmentation_kwargs": {"f":10, "mask_area": mask_area}, # factor to standard deviation, and the specific area wheredetection is allowed
                  "mask_filter": fitler_objects_size, # second filtering step that removes objects according to their size
                 "mask_filter_kwargs":{"min_size": 8}, # minimal size of objects  #### 5 in original ants tracker
                 "detection": simple_detection} # detection by weighted centroid
### I see some ants are detected twice/ but this is probabaly not a big deal

# tracking and stitching parameters
max_tracking_dist = 100 # original was 150
min_frame_dist_stitching = -2
max_frame_dist_stitching = 8
max_dist_stitching = 30 # very low number
min_track_length = 5


# database properties
markers = {"detections":"#00FF00"}
masks =  {"ants":["#0000FF",1]}
layers =  ["images"]
new_folder = None





db_name = "database.cdb" # databesename
images = [x for x in os.listdir(folder) if re.search(im_selector, x)]
images = sorted(images, key=lambda x: int(re.search(im_selector,x).group(1))) # hoep that works always

# setting up the data base and adding images
createFolder(outputfolder)
db = setup_masks_and_layers(db_name, folder, outputfolder, markers, masks, layers)
add_images(db, images)

# detection
print('-->Detection')
for frame in tqdm(range(db.getImageCount() - 1)):
    cdb_add_detection(frame, db, detect, cdb_types="detections", detct_kwargs=detect_kwargs)

# tracking
print('-->Tracking')
tracking_opt(db, max_tracking_dist, type="detections", color="#00FF00")
# copy database before stitching
copyfile(os.path.join(outputfolder, db_name), os.path.join(outputfolder, "not_stitched" + db_name))

print('-->Reading tracks from data base')
tracks_dict = tracks_to_dict(db, t_type="trackdetections")
print('-->Stitiching')
tracks_stitched1, stitched_id, gaps, old_ids = stitch(tracks_dict, f_min=min_frame_dist_stitching
                                                      , f_max=max_frame_dist_stitching, s_max=max_dist_stitching,
                                                      method="sparse")
# filtering small tracks:
tracks_stitched1 = {key: value for key, value in tracks_stitched1.items() if len(value) > min_track_length}

#### removes all exisitng markers from the database!!!!!!!!!!
db.deleteMarkers()
db.deleteTracks()

write_tracks_dict_to_db(db, tracks_stitched1, "trackdetections")

