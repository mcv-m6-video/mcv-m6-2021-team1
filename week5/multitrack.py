import os
import sys
import cv2
import numpy as np
import utils


def match_tracks(query, query_cam, candidates, candidates_cam):
    print(f'We are looking for a match between the id {query} in the camera {query_cam} and the list {candidates} in {candidates_cam}.')
    return list(candidates)[0], 1

###CONGIF###

DATA_PATH = 'C:\\Users\\Carmen\\CVMaster\\M6\\aic19-track1-mtmc-train'
DATA_PATH = '/home/capiguri/code/datasets/m6data/'
SEQ = 1

############

FRAME_NUM_PATH = os.path.join(DATA_PATH, 'cam_framenum')
TIMESTAMP_PATH = os.path.join(DATA_PATH, 'cam_timestamp')
TRACK_PATH = os.path.join(DATA_PATH, f'train', f'S0{SEQ}')

#Load individual camera trackings
num_cams = sum(1 for line in open(os.path.join(FRAME_NUM_PATH, f'S0{SEQ}.txt')))
dic_tracks_byframe = [utils.parse_aicity_rects(os.path.join(TRACK_PATH, f'c{str(cam).zfill(3)}','gt', 'gt.txt')) for cam in range(1, num_cams+1)]

##TODO: Load Neighbourhood
adj_mat = np.array([[0, 1, 1, 1, 1],
                    [1, 0, 1, 1, 1],
                    [1, 1, 0, 1, 1],
                    [1, 1, 1, 0, 1],
                    [1, 1, 1, 1, 0]])
adj_mat = np.triu(adj_mat)

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

time_stamps = []
with open(os.path.join(TIMESTAMP_PATH, f'S0{SEQ}.txt')) as f:
    for line in f:
        time_stamps.append(float(line.split()[-1].replace('\n','')))

max_frames = []
with open(os.path.join(FRAME_NUM_PATH, f'S0{SEQ}.txt')) as f:
    for line in f:
        max_frames.append(int(line.split()[-1].replace('\n','')))

mt_cont = 0
for cam in range(0, num_cams):

    ##Update mt for myself
    for key_query, element_query in dic_tracks[cam].items():
        for fr_instance in element_query:
            for el_t in dic_tracks_byframe[cam][f'f_{fr_instance}']:
                if el_t['mt_id'] == -1:
                    el_t['mt_id'] = mt_cont
        mt_cont+=1
    
    for key_query, element_query in dic_tracks[cam].items():
        fr_min = np.min(list(element_query.keys()))
        fr_max = np.max(list(element_query.keys()))

        for i in range(len(adj_mat[cam])):

            if adj_mat[cam][i]: #if this cam is a neighbour of the query

                # Account for time mistmach
                offset = int(time_stamps[i]*10) # 10 FPS
                fr_min_cand = fr_min - offset if fr_min > offset else 0
                fr_max_cand  = fr_max - offset if fr_max > offset else 0

                # Increase search margin
                extension = 2*10 # how many seconds (by frame rate)
                fr_min_cand = fr_min_cand-extension if fr_min_cand>extension else 0
                fr_max_cand = fr_max_cand+extension if fr_max_cand+extension<max_frames[i] else max_frames[i]

                candidates=set()
                for f_cand in range(fr_min_cand, fr_min_cand+1):
                    if f'f_{f_cand}' in dic_tracks_byframe[i]:
                        for obj_cand in dic_tracks_byframe[i][f'f_{f_cand}']:
                            candidates.add(obj_cand['id'])

                match, conf = match_tracks(key_query, cam, candidates, i) if candidates else (-1, 1)
        
                for k_mt, el_mt in dic_tracks_byframe[i].items():
                    for obj_el in el_mt:
                        if obj_el['id'] == match and not(obj_el['mt_id'] != -1 and obj_el['mt_conf']>conf):
                            fr_aux = list(dic_tracks[cam][key_query].keys())[0]

                            for ei in dic_tracks_byframe[cam][f'f_{fr_aux}']:
                                if ei['id'] == key_query:
                                    obj_el['mt_id'] = ei['mt_id']
                            obj_el['mt_conf'] = conf


for i, det in enumerate(dic_tracks_byframe):
    utils.save_aicity_rects(f'mtmc_S{str(SEQ).zfill(2)}_c{str(i).zfill(3)}_random.txt', det, True)