# coding: utf-8
# author: Christoph Mark
# version: 2018-09-05

from tqdm import tqdm
import clickpoints
import os


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


def setup_masks_and_layers(db_name, folder, markers, masks, layers):

    cdb_filepath = os.path.join(folder, db_name)
    db = clickpoints.DataFile(cdb_filepath, 'w')  # creates and opens the cdb file
    db.setPath(folder, 1)  # sets path entry of images in cdb file

    # setting up marker
    for name, color in markers.items():
        db.setMarkerType(name=name, color=color)

    # setting up mask types
    for name, (color,index) in masks.items():
        db.setMaskType(name=name, color=color, index=index)

    # setting up layers
    base_layer = db.getLayer(layers[0], create=True, id=0)
    for name in layers[1:]:
        db.getLayer(name, base_layer=base_layer, create=True)



    return db

def add_images(db, images):
    # images must contain full paths
    for i, file in tqdm(enumerate(images), total=len(images)):
        image = db.setImage(filename=os.path.split(file)[1], path=1, layer="images", sort_index=i)

