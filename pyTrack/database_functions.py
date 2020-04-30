# coding: utf-8
# author: Christoph Mark
# version: 2018-09-05
from pyTrack.utilities import *
from tqdm import tqdm
import clickpoints
import os
from collections import defaultdict
import traceback
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, select, event


def annotate(db):
    '''
    parameters:
    db-any cdb database object

    Annotates Start and end of tracks. Use this to evaluate stiching and general tracking.
    '''
    for track in tqdm(db.getTracks()):
        if len(track.markers)==2:
            start_end = (track.markers[0], track.markers[-1])  # gets start and end marker
            track_id=track.id
            type=track.type
            db.deleteMarkers(id=[track.markers[0].id, track.markers[-1].id]) # deletes original markers
            # restore track(old track is deleted automatically)
            track=db.setTrack(type='track',id=track_id)
            # sets new marker with text at start
            db.setMarker(x=start_end[0].x, y=start_end[0].y, text="track_start" + "_id" + str(track_id),
                         frame=start_end[0].image.id - 1, type=type, track=track)
            # sets new marker with text at end
            db.setMarker(x=start_end[1].x, y=start_end[1].y, text= "track_end" + "_id" + str(track_id),
                         frame=start_end[1].image.id - 1, type=type, track=track)

        if len(track.markers)>2:
            start_end = (track.markers[0], track.markers[-1])  # gets start and end marker
            track_id=track.id
            db.deleteMarkers(id=[track.markers[0].id, track.markers[-1].id]) # deletes original markers

            # sets new marker with text at start
            db.setMarker(x=start_end[0].x, y=start_end[0].y, text="track_start" + "_id" + str(track_id),
                         frame=start_end[0].image.id - 1, type=track.type, track=track)
            # sets new marker with text at end
            db.setMarker(x=start_end[1].x, y=start_end[1].y, text= "track_end" + "_id" + str(track_id),
                         frame=start_end[1].image.id - 1, type=track.type, track=track)


def setup_masks_and_layers(db_name, input_folder, output_folder, markers=None, masks=None, layers=None, tracks=None):

    cdb_filepath = os.path.join(output_folder, db_name)
    db = clickpoints.DataFile(cdb_filepath, 'w')  # creates and opens the cdb file
    db.setPath(input_folder, 1)  # sets path entry of input images in cdb file
    #db.setPath(output_folder, 2)  # sets path entry of outputfolder images in cdb file
    # setting up marker types
    if not markers is None:
        for name, color in markers.items():
            db.setMarkerType(name=name, color=color)


    # setting up track types
    if not tracks is None:
        for name, color in tracks.items():
            db.setMarkerType(name=name, color=color, mode=db.TYPE_Track)

    # setting up mask types
    if not masks is None:
        for name, (color, index) in masks.items():
            db.setMaskType(name=name, color=color, index=index)

    # setting up layers
    if not layers is None:
        base_layer = db.getLayer(layers[0], create=True, id=0)
        for name in layers[1:]:
            db.getLayer(name, base_layer=base_layer, create=True)

    return db

def add_images(db, images):
    # images must contain full paths
    for i, file in tqdm(enumerate(images), total=len(images)):
        image = db.setImage(filename=os.path.split(file)[1], path=1, layer="images", sort_index=i)

class OpenDB:
    # context manager for database file. if db is a path a new file handle is opened
    # this handle is later closed (!). if db is already clickpoints.DataFile object,
    # the handle is not closed
    def __init__(self,db, method="r", raise_Error=True):
        self.raise_Error = raise_Error
        if isinstance(db, clickpoints.DataFile):
            self.file = db
            self.db_obj=True
        else:
            self.file = clickpoints.DataFile(db, method)
            self.db_obj = False

    def __enter__(self):
        return self.file

    def __exit__(self, exc_type, exc_value, trace):

        if not self.db_obj:
            self.file.db.close()
        if self.raise_Error:
            return False
        else:
            traceback.print_tb(trace)
            return True




def read_markers_iteratively(db, marker_types=None, start_frame=0, end_frame=None, mode="all"):
    # sort_id_mode == 1 identifies the frames(sort_indices) where the markers are directly
    # sort_id_mode == 2 assumes sort_index=image_id in the database, this is 3 times faster and generally applies if you area
    # not dealing with multiple layers or separetly added images

    # track_type=None for all tracks
    tracks_dict = defaultdict(list)  ### improve by using direct sql query or by iterating through frames
    if not marker_types is None:
        marker_types = make_iterable(marker_types)


    with OpenDB(db) as db_l:

        if end_frame is None:
           end_frame =  db_l.getImageCount()

        marker_type_ids = []
        if not marker_types is None:
            for marker_type in marker_types:
                if isinstance(marker_type, str):
                    t_type_id = db_l.db.execute_sql(
                        "SELECT id FROM markertype WHERE name = '%s'" % (marker_type)).fetchall()
                    if len(t_type_id) == 0:
                        print("couldn't find marker/track type " + marker_type)
                    else:
                        marker_type_ids.append(t_type_id[0][0])
                else:
                    marker_type_ids.append(marker_type)


        # image id: sort id
        sort_id_dict = {x[0]: x[1] for x in db_l.db.execute_sql(
            "SELECT id, sort_index FROM image WHERE  (sort_index >= %s AND sort_index < %s)" % (
                str(start_frame), str(end_frame))).fetchall()}


        if mode=="all_frames":
            track_type_selector = " WHERE type_id IN (%s)" % ", ".join([str(x) for x in marker_type_ids]) if not marker_types is None else ""
            q = db_l.db.execute_sql("SELECT x, y, image_id FROM marker" + track_type_selector)

        else:
            track_type_selector = " AND type_id IN (%s)" % ", ".join(
                [str(x) for x in marker_type_ids]) if not marker_types is None else ""
            # all relevant image ids
            image_id_list = "(%s)" % ", ".join([str(x) for x in list(sort_id_dict.keys())])
            q = db_l.db.execute_sql(
                "SELECT x, y, image_id FROM marker WHERE image_id IN %s%s" % (
                image_id_list, track_type_selector))
        markers_dict = defaultdict(list) #frame:[x,y]
        for m in tqdm(q.fetchall()):
            markers_dict[sort_id_dict[m[2]]].append([m[0], m[1],])

        return markers_dict



def read_tracks_iteratively(db, track_types=None, start_frame=0, end_frame=None, sort_id_mode=2):
    # sort_id_mode == 1 identifies the frames(sort_indices) where the markers are directly
    # sort_id_mode == 2 assumes sort_index=image_id in the database, this is 3 times faster and generally applies if you area
    # not dealing with multiple layers or separetly added images

    # track_type=None for all tracksn
    tracks_dict = defaultdict(list)  ### improve by using direct sql query or by iterating through frames
    if not track_types is None:
        track_types = make_iterable(track_types)


    with OpenDB(db) as db_l:

        if end_frame is None:
           end_frame =  db_l.getImageCount()

        track_type_ids = []
        if not track_types is None:
            for track_type in track_types:
                if isinstance(track_type, str):
                    t_type_id = db_l.db.execute_sql(
                        "SELECT id FROM markertype WHERE name = '%s'" % (track_type)).fetchall()
                    if len(t_type_id) == 0:
                        print("couldn't find marker/track type " + track_type)
                    else:
                        track_type_ids.append(t_type_id[0][0])
                else:
                    track_type_ids.append(track_type)
            track_type_selector = " AND type_id IN (%s)" % ", ".join(
                [str(x) for x in track_type_ids]) if not track_types is None else ""




        if sort_id_mode == 1:
            # image id: sort id
            sort_id_dict = {x[0]: x[1] for x in db_l.db.execute_sql(
                "SELECT id, sort_index FROM image WHERE  (sort_index >= %s AND sort_index < %s)" % (
                str(start_frame), str(end_frame))).fetchall()}
            # all relevant image ids
            image_id_list = "(%s)" % ", ".join([str(x) for x in list(sort_id_dict.keys())])
            q = db_l.db.execute_sql(
                "SELECT x, y, image_id, track_id FROM marker WHERE image_id IN %s%s" % (
                image_id_list, track_type_selector))

        if sort_id_mode == 2:
            # image id: sort id
            sort_id_dict = {x[0]: x[1] for x in db_l.db.execute_sql(
                "SELECT id, sort_index  FROM image WHERE  (sort_index >= %s AND sort_index < %s)" % (
                str(start_frame), str(end_frame))).fetchall()}
            q = db_l.db.execute_sql(
                "SELECT x, y, image_id, track_id FROM marker WHERE (image_id >= %s AND image_id < %s)%s" % (
                str(start_frame), str(end_frame), track_type_selector))

        for m in tqdm(q.fetchall()):
            tracks_dict[m[3]].append([m[0], m[1],sort_id_dict[m[2]]])
        #tracks_dict = {t_id: np.array(v) for t_id, v in tracks_dict.items()} # conversion to array// maybe include in loop above
    return dict(tracks_dict) # conversion to normal dict// otherwise empty keys are problematic

'''
import time
t1=time.time()
read_tracks_iteratively(db, track_types=None, start_frame=0, end_frame=300)
t2=time.time()
print(t2-t1)

t1=time.time()
read_tracks_iteratively2(db, track_types=None, start_frame=0, end_frame=300)
t2=time.time()
print(t2-t1)
'''

def write_tracks_dict_to_db(db, tracks_dict, marker_type, method=2):

    # method1: 100 times faster then the others, but doesn't insert the markers correctly--> might produce problems when
    # in the database structure
    # mehtod2: pure clickpoints API mehtof
    # method3: significantly (?) faster then method2 --> use this in general?

    # finding highest existing id
    old_tracks = db.getTracks()
    if len(old_tracks) > 0 :
        max_track_id = db.getTracks()[-1].id + 1
    else:
        max_track_id=0
    # constructing new list of track ids. Note this reassignes the ids starting from the lowest existing track id + 1
    track_ids = {t_id_old: t_id_new + max_track_id for t_id_new, t_id_old in enumerate(tracks_dict.keys())}
    # creating all new tracks at once
    ttype = db._processesTypeNameField(marker_type, ["TYPE_Track"])
    m_id = ttype.id
    parameters= [{"id": t_id , "type": ttype} for t_id in track_ids.values()]
    db.saveReplaceMany(db.table_track, parameters)



    sort_id_dict = {x[0]: x[1] for x in db.db.execute_sql("SELECT sort_index, id FROM image").fetchall()}
   # m=db.setMarker(frame=1,x=0,y=0,type=ttype)
    old_markers = db.getMarkers()
    if len(old_markers) > 0:

        max_marker_id = db.getMarkers()[-1].id + 1
    else:
        max_marker_id = 0

    id_list = []
    image_id_list = []
    x_list = []
    y_list = []
    frame_list=[]
    type_id_list = []
    type_str_list = []# not sure what that is
    processed_list = []
    track_id_list = []
    style_list = []
    text_list = []
    for t_id_old, xyf in tracks_dict.items():
        xyf = np.array(list(xyf))
        xyf = np.expand_dims(xyf,axis=0) if xyf.ndim==1 else xyf # making sure that the dimesions of the track array are correct

        new_ids = list(range(max_marker_id,max_marker_id + len(xyf)))
        max_marker_id+=len(xyf)
        id_list.extend(new_ids)

        image_id_list.extend([sort_id_dict[int(frame)] for frame in xyf[:,2]])
        x_list.extend(list(xyf[:,0]))
        y_list.extend(list(xyf[:,1]))
        frame_list.extend(list(xyf[:,2].astype(int)))
        type_id_list.extend([m_id]*len(xyf))
        type_str_list.extend([ttype]*len(xyf))
        processed_list.extend([0]*len(xyf))
        track_id_list.extend([track_ids[t_id_old]]*len(xyf))
        style_list.extend([None]*len(xyf))
        text_list.extend(["track_" + str(track_ids[t_id_old])]*len(xyf))

    #insert_values = list(zip(image_id_list , x_list , y_list , type_id_list, processed_list , track_id_list, style_list, text_list))

    if method==1:
        insert_values = list(
            zip(id_list, image_id_list, x_list, y_list, type_id_list, processed_list, track_id_list, style_list,
                text_list))
        # still to slow???--> could be improved by first writing to csv file
        engine = create_engine("sqlite:////" + db._database_filename)
        engine.connect()
        block_size = 10000
        for s in tqdm(range(0, len(image_id_list), block_size)):
            block_size = 10000
            # @event.listens_for(engine, 'before_cursor_execute')
            # def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
            #    print("FUNC call")
            #    if executemany:
            #       cursor.fast_executemany = True

            df = pd.DataFrame(insert_values[s:s + block_size],
                              columns=["id", "image_id", "x", "y", "type_id", "processed", "track_id", "style", "text"])
            df.to_sql("marker", engine, if_exists="append", chunksize=None, index=False)


    if method==2:
        block_size = 10000
        for s in tqdm(range(0, len(image_id_list), block_size)):
            db.setMarkers(frame=frame_list[s:s + block_size],x=x_list[s:s + block_size],y=y_list[s:s + block_size],type=type_str_list[s:s + block_size],track=track_id_list[s:s + block_size])

    if method == 3:
        block_size = 10000
        cur = db.db.cursor()
        insert_values = list(
            zip(id_list, image_id_list, x_list, y_list, type_id_list, processed_list, track_id_list, style_list,
                text_list))
        query = "insert into marker (id, image_id, x, y, type_id, processed, track_id, style, text) values (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        for s in tqdm(range(0, len(image_id_list), block_size)):
            try:
                cur.executemany(query, insert_values[s:s + block_size])
                db.db.commit()
            except Exception as e:
                db.db.rollback()
                raise e



