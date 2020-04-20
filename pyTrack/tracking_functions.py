
from collections import defaultdict
import numpy as np
from pyTrack.utilities import *
import shutil

def tracking_clickpoints(sort_index, db, r):
    ##with weighted_centroid

    '''
    parameters:
    sort_index- number of the frame of a cdb image
    db- a cdb database objec
    r- maximal radius to allow assignement of new detection to a track


    This function joins detections to the closest track, if the track is within a maximal distance. If no track can be
    associated, as well as for the first frame a new track object is created. The individual detections are transfered
    to the new type "detections
    '''


    detections = db.getMarkers(frame=sort_index, type='detection_prelim')[:]

    # first time step: all detections initiate new tracks
    if sort_index == 0:
        for marker in detections:
            # set new track_type marker
            new_track=db.setTrack('track')
            db.setMarker(frame=sort_index, type='track', x=marker.x, y=marker.y, track=new_track,
                         text="track_" + str(new_track.id))
            # transfering old marker to new type
            marker.changeType('detection')

    # after first step: assign detections to tracks or generate new tracks
    else:
        # get all existing tracks from the previous frame
        tracks = db.getMarkers(frame=sort_index-1, type='track')[:]
        # measure distances between current detections and dections from the previous frame

        distance_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                track_position = np.array([tracks[i].x, tracks[i].y])
                detected_position = np.array([detection.x, detection.y])
                distance = np.sqrt(np.sum((detected_position - track_position)**2.))
                distance_matrix[i, j] = distance #constructing a distance matrix

        # assign nearest detections to track (if distance is below search_radius in pixels)
        while True:
            try:
                [i, j] = ndargmin(distance_matrix)   # custom function, returns idnex of closest pair
                distance = distance_matrix[i, j]
                if distance > r:  # no more suitable candidates to assign
                    break
            except ValueError:
                break  # only NaN entries left, nothing to assign

            # add corresponing marker to track
            db.setMarker(frame=sort_index, type='track',
                         x=detections[j].x, y=detections[j].y,
                         track=tracks[i].track, text="track_"+ str(tracks[i].track.id))
            # transfering individual detection to a new type
            detections[j].changeType('detection')

            distance_matrix[i, :] = np.nan  # fill columns and rows, belonging to connected tracks points with nans
            distance_matrix[:, j] = np.nan

        # for remaining detections, create new tracks
        for marker in db.getMarkers(frame=sort_index, type='detection_prelim'):
            new_track=db.setTrack('track')
            db.setMarker(frame=sort_index, type='track',
                         x=marker.x, y=marker.y,
                         track=new_track, text="track_" + str(new_track.id))
            marker.changeType('detection')


def tracking_beads_clickpoints(sort_index, db, r):
    '''
    parameters:
    sort_index- number of the frame of a cdb image
    db- a cdb database objec
    r- maximal radius to allow assignement of new detection to a track


    This function joins detections to the closest track, if the track is within a maximal distance. If no track can be
    associated, as well as for the first frame a new track object is created. The individual detections are transfered
    to the new type "detections
    '''

    all_markers = db.getMarkers(frame=sort_index, type=["cells_without_beads_prelim", 'cells_with_beads_prelim'])[:]
    detections_beads = db.getMarkers(frame=sort_index, type='cells_with_beads_prelim')[:]
    detections_without_beads = db.getMarkers(frame=sort_index, type='cells_without_beads_prelim')[:]

    # first time step: all detections become tracks

    if sort_index == 0:
        for marker in detections_beads:
            # set new track_type marker
            new_track = db.setTrack('track')
            db.setMarker(frame=sort_index, type='track', x=marker.x, y=marker.y, track=new_track,
                         text="track_" + str(new_track.id))
            # transfering old marker to new type
            marker.changeType('cells_with_beads')

        for marker in detections_without_beads:
            # set new track_type marker
            new_track = db.setTrack('track')
            db.setMarker(frame=sort_index, type='track', x=marker.x, y=marker.y, track=new_track,
                         text="track_" + str(new_track.id))
            # transfering old marker to new type
            marker.changeType('cells_without_beads')


    # after first step: assign detections to tracks or generate new tracks
    else:
        # get all existing tracks from the previous frame
        tracks = db.getMarkers(frame=sort_index-1, type='track')[:]
        # measure distances between current detections and dections from the previous frames
        distance_matrix = np.zeros((len(tracks), len(all_markers)))
        for i, track in enumerate(tracks):
            for j, detection in enumerate(all_markers):
                track_position = np.array([tracks[i].x, tracks[i].y])
                detected_position = np.array([detection.x, detection.y])
                distance = np.sqrt(np.sum((detected_position - track_position)**2.))
                distance_matrix[i, j] = distance #constructing a distance matrix

        # assign nearest detections to track (if distance is below search_radius in pixels)
        while True:
            try:
                [i, j] = ndargmin(distance_matrix)   # custom function, returns idnex of closest pair
                distance = distance_matrix[i, j]
                if distance > r:  # no more suitable candidates to assign
                    break
            except ValueError:
                break  # only NaN entries left, nothing to assign

            # add corresponing marker to track
            db.setMarker(frame=sort_index, type='track',
                         x=all_markers[j].x, y=all_markers[j].y,
                         track=tracks[i].track,text="track_"+ str(tracks[i].track.id))

            # transfering individual detection to a new type
            if all_markers[j].type.name=='cells_with_beads_prelim':
                all_markers[j].changeType('cells_with_beads')
            if all_markers[j].type.name == 'cells_without_beads_prelim':
                all_markers[j].changeType('cells_without_beads')

            distance_matrix[i, :] = np.nan  # fills columns and rows, belonging to  connected tracks points with zeros
            distance_matrix[:, j] = np.nan

        for marker in db.getMarkers(frame=sort_index, type='cells_with_beads_prelim'):
            new_track = db.setTrack('track')
            db.setMarker(frame=sort_index, type='track', x=marker.x, y=marker.y,track=new_track ,
                         text="track_" + str(new_track.id))
            marker.changeType('cells_with_beads')

        for marker in db.getMarkers(frame=sort_index, type='cells_without_beads_prelim'):
            new_track = db.setTrack('track')
            db.setMarker(frame=sort_index, type='track', x=marker.x, y=marker.y, track=new_track ,
                         text="track_" + str(new_track.id))
            marker.changeType('cells_without_beads')

    return

def nearest_neighbour_assignement(det1, det2, max_dist):
    '''
    assining position in det1 to positions of det2 by nearest neighbour algorithm
    :param det1:
    :param det2:
    :param max_dist:
    :return:
    '''

    distances = np.linalg.norm(det1[None, :] - det2[:, None], axis=2)
    assigne={} # assignment key: value --> detection id current frame:detection id previous frame
    while np.nanmin(distances) < max_dist:
        min_pos = list(np.unravel_index(np.nanargmin(distances), distances.shape))
        distances[min_pos[0], :] = np.nan
        distances[:, min_pos[1]] = np.nan
        assigne[int(min_pos[0])] = int(min_pos[1])
    return assigne


class Track():
    def __init__(self,out_file):
        self.out_file = out_file
        self.tracks_dict = {} # frame: detection_id: [track_id,np.array([x,y])]
        self.max_track_id = 0
    def add_frame(self,frame,detections,max_dist,first=False):
        if first:
            self.tracks_dict[frame]={i:[i,det] for i,det in enumerate(detections)}
        else:
            det1= np.array([d[1] for d in self.tracks_dict[frame-1]])
            assign = nearest_neighbour_assignement(det1, detections, max_dist)

            # iterating through all detections in current frame
            for d_id,det in enumerate(detections):
                # grabbing the track of the detection from the previous frame
                if d_id in assign.keys(): # if curr detection was assigned to prev detection
                    # notes index to  track of previous detection to new detection
                    self.tracks_dict[frame][d_id] = self.tracks_dict[frame - 1][assign[d_id]]
                # assigning a new tack id
                else:
                    self.max_track_id = 1 +  self.max_track_id
                    self.tracks_dict[frame][d_id] =  [self.max_track_id,det]  # assigning new id if not assigned to old one

    def dump_frame(self,file):
        # sequential writing to text file, can be used for long experiments// could be much more efficient i guess

        self.tracks_by_id() # making a dictionary t_id: frames: [x,y] for easier writing
        self.write_tracks_minimal(self,file) # write to file by appending to rows
        # emptying self.tracks_dict except for last frame
        all_frames = list(self.tracks_dict.keys())
        max_frame = np.max(all_frames)
        for frame in all_frames:
            if frame!=max_frame:
                del self.tracks_dict[frame]


    def tracks_by_id(self):
        self.tracks_dict_rev=defaultdict(dict) # t_id: frames: [x,y]
        for frame,sub_dict in self.tracks_dict.items():
            for d_id,[t_id,det] in sub_dict.items():
                self.tracks_dict_rev[t_id][frame]=det
        return self.tracks_dict_rev


    def write_tracks_minimal(self,file):
        # appending lines if new frames have been found, could take
        temp_path=os.path.join(os.path.split(file)[0],"_temp_out.txt")
        with open(file,"r") as f, open(temp_path,"w") as f_temp:
            for line in f:
                l1 = line.split("\t")
                frames = [int(x.strip().split(",[")[0]) for x in l1[1:]]
                t_id = int(l1[0])
                if t_id in self.tracks_dict_rev.keys():
                    new_l = "\t".join([str(f) + "," + str(det) for f, det in self.tracks_dict_rev[t_id].items() if f not in frames])
                    line = line.strip() + "\t" + new_l + "\n" if len(new_l)>0 else line

                f_temp.write(line)

        shutil(f_temp,file)





def nearest_neighbour_tracking(det1_ids, det2_ids, n_det2_ids, frame, max_track_id, tracks):
    '''

    :param det1_ids: ids (positions in the list) of detections1 --> from previous frame
    :param det2_ids: ids (positions in the list) of detections2 --> from current frame
    :param n_det2_ids: length of detections2
    :param max_track_id: maximum
    :return:

    '''

    # iterating through all #indices of new detections
    pass

def tracking_opt(db, max_dist, type="",color="#0000FF"):


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

    for frame in tqdm(range(1, db.getImageCount() - 1)):
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
        new_track = db.setTrack('track'+type)
        xs = [x[1][0] for x in values]
        ys = [x[1][1] for x in values]
        frames = [x[0] for x in values]
        db.setMarkers(frame=frames, type='track'+type, x=xs, y=ys, track=new_track,
                      text="track_" + str(id))
