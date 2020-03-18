import numpy as np

def write_tracks(tracks, detections, file, frame, max_track_id):
    '''
    writes tracks to text file. First col gives frame Id. For each other column
    the number of the coulmn gives the track id. Detections of the tracks are
    written as x,y positions in each cell. If the entry is -1,-1 then no detection
    was assigned in this frame to this track
    :param: tracks, dictionary with index_of_detection: track_id
    :param: detection, list of x,y positions of the detections in one frame
    :param: file, filename of a text file, either existing or to be generated
    :param:frame, number of the current frame
    :param: max_track_id, number of the highest current track id
    '''
    # file has no explicit header, but the track id is the col index,
    # except for the first column. First col is frame id.

    track_ids = np.array(list(tracks.values()), dtype=int)  # track ids
    detect_ids = np.array(list(tracks.keys()), dtype=int)  # index of the detection in this assignement
    max_track_id = np.max(track_ids)
    line_x = np.zeros(int(max_track_id + 1)) - 1
    line_y = np.zeros(int(max_track_id + 1)) - 1

    line_x[track_ids] = detections[detect_ids, 0]
    line_x = line_x.astype(np.unicode_).tolist()
    line_y[track_ids] = detections[detect_ids, 1]
    line_y = line_y.astype(np.unicode_).tolist()
    line = [x + "," + y for x, y in zip(line_x, line_y)]  # writing as x,y values in one line
    line = [str(frame)] + line

    line = "\t".join(line) + "\n"  # seperating by tab
    with open(file, "a+") as f:
        f.write(line)



