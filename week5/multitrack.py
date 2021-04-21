import os
import sys
import cv2
import numpy as np
import utils

###CONGIF###

DATA_PATH = 'C:\\Users\\Carmen\\CVMaster\\M6\\aic19-track1-mtmc-train'
SEQ = 1

############

FRAME_NUM_PATH = os.path.join(DATA_PATH, 'cam_framenum')
TIMESTAMP_PATH = os.path.join(DATA_PATH, 'cam_timestamp')
TRACK_PATH = os.path.join(DATA_PATH, f'train\\S0{SEQ}')

#Load individual camera trackings
num_cams = sum(1 for line in open(os.path.join(FRAME_NUM_PATH, f'S0{SEQ}.txt')))
dic_tracks_byframe = [utils.parse_aicity_rects(os.path.join(TRACK_PATH, f'c00{cam}\\gt\\gt.txt')) for cam in range(1, num_cams+1)]

##TODO: Load Neighbourhood
adj_mat = np.array([[0, 1, 1, 1, 1],
                    [1, 0, 1, 1, 1],
                    [1, 1, 0, 1, 1],
                    [1, 1, 1, 0, 1],
                    [1, 1, 1, 1, 0]])

#Sort by id
dic_tracks = []
for cam in range(0, num_cams):
    dic_aux = {}
    for key, frame in dic_tracks_byframe[cam].items():
        for obj_t in frame:
            if obj_t['id'] not in dic_aux:
                dic_aux[obj_t['id']] = {}
            dic_aux[obj_t['id']][int(key[2:])] = {'bbox': obj_t['bbox'], 'conf': obj_t['conf']}        
    dic_tracks.append(dic_aux)

print(dic_tracks[0][69].keys())

time_stamps = []
with open(os.path.join(TIMESTAMP_PATH, f'S0{SEQ}.txt')) as f:
    for line in f:
        time_stamps.append(float(line.split()[-1].replace('\n','')))

max_frames = []
with open(os.path.join(FRAME_NUM_PATH, f'S0{SEQ}.txt')) as f:
    for line in f:
        max_frames.append(int(line.split()[-1].replace('\n','')))

for cam in range(0, num_cams):
    for key_query, element_query in dic_tracks[cam].items():
        fr_min = np.min(element_query.keys())
        fr_max = np.max(element_query.keys())
        
        for i in range(adj_mat[cam]):
            if adj_mat[cam][i]: #if this cam is a neighbour of the query
                # Account for time mistmach
                offset = int(time_stamps[i]*10) # 10 FPS
                fr_min_cam = fr_min - offset if fr_min > offset else 0
                fr_max_cam  = fr_max - offset if fr_max > offset else 0

                # Increase search margin
                extension = 10 # how many frames
                fr_min_cam -= extension if fr_min_cam > extension else 0
                fr_max_cam  += extension if fr_max + extension < max_frames[i] else max_frames[i]


            # dic_tracks_byframe[i]
