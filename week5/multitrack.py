import os
import sys
import cv2
import numpy as np
import utils
from matplotlib import pyplot as plt

###CONGIF###

DATA_PATH = 'C:\\Users\\Carmen\\CVMaster\\M6\\aic19-track1-mtmc-train'
# DATA_PATH = '/home/capiguri/code/datasets/m6data/'
SEQ = 1
METHOD = 'hist_rgb'

############

FRAME_NUM_PATH = os.path.join(DATA_PATH, 'cam_framenum')
TIMESTAMP_PATH = os.path.join(DATA_PATH, 'cam_timestamp')
TRACK_PATH = os.path.join(DATA_PATH, f'train', f'S0{SEQ}')

CAM_NAMES = []
with open(os.path.join(TIMESTAMP_PATH, f'S0{SEQ}.txt')) as f:
    for line in f:
        CAM_NAMES.append(line.split()[0])

def crop_bbox(cam, frame, bbox):
    cap_aux = cv2.VideoCapture(os.path.join(DATA_PATH, 'train', f'S{str(SEQ).zfill(2)}', CAM_NAMES[cam], 'vdo.avi'))
    cap_aux.set(1, frame)
    _, fr_im = cap_aux.read()
    cap_aux.release()
    return fr_im[bbox[1]:bbox[3], bbox[0]:bbox[2]]

def hist_rgb_match(query_data, cand_data):

    query_im = crop_bbox(*query_data[0])
    cand_im = crop_bbox(*cand_data[0])

    H1 = np.vstack([cv2.calcHist([query_im],[i],None,[256],[0,256]) for i in range(3)])
    H2 = np.vstack([cv2.calcHist([cand_im],[i],None,[256],[0,256]) for i in range(3)])

    H1 = cv2.normalize(H1, H1, norm_type=cv2.NORM_L2)
    H2 = cv2.normalize(H2, H2, norm_type=cv2.NORM_L2)

    conf = 1 - cv2.compareHist(H1, H2, cv2.HISTCMP_INTERSECT)

    return conf

def hist_3d_match(query_data, cand_data):

    query_im = crop_bbox(*query_data[0])
    cand_im = crop_bbox(*cand_data[0])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # preferential number of bins for each channel based on experimental results
    hist = cv2.calcHist([image], [0, 1, 2], mask, [int(bins/4), 3*bins, 3*bins], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist)
    hist.flatten()

    conf = cv2.compareHist(H1, H2, cv2.HISTCMP_HELLINGER)

    return conf

def match_tracks(query, query_cam, candidates, candidates_cam, dic_data, method):
    # print(f'We are looking for a match between the id {query} in the camera {query_cam} and the list {candidates} in {candidates_cam}.')

    query_data = [(query_cam, fr_query, dic_data[query_cam][query][fr_query]['bbox']) 
        for fr_query in dic_data[query_cam][query]]

    best_cand = -1
    best_cand_conf = 0
    for cand in candidates:
        # print('cand:', cand)
        cand_data = [(candidates_cam, fr_cand, dic_data[candidates_cam][cand][fr_cand]['bbox']) 
            for fr_cand in dic_data[candidates_cam][cand]]
    
        if method == 'hist_rgb':
            conf = hist_rgb_match(query_data, cand_data)
        elif method == 'hist_3d':
            conf = hist_3d_match(query_data, cand_data)
        else:
            continue

        if conf>best_cand_conf:
            best_cand = cand
            best_cand_conf = conf

    cv2.destroyAllWindows()
    # print(f'Best match selected: {best_cand} with conf: {best_cand_conf}')
    return best_cand, best_cand_conf
        
# def match_tracks(query, query_cam, candidates, candidates_cam, dic_data, method):
    
#     fr_query = list(dic_data[query_cam][query].keys())[0]
#     bb_query = dic_data[query_cam][query][fr_query]['bbox']

#     query_im = crop_bbox(query_cam, fr_query, bb_query)

#     # cv2.imshow('im_query', query_im)
#     # cv2.waitKey(0)

#     best_cand = -1
#     best_cand_conf = 0
#     for cand in candidates:

#         fr_cand = list(dic_data[candidates_cam][cand].keys())[0]
#         bb_cand = dic_data[candidates_cam][cand][fr_cand]['bbox']

#         cand_im = crop_bbox(candidates_cam, fr_cand, bb_cand)

#         # cv2.imshow(f'candidate {cand}', cand_im)
#         # cv2.waitKey(0)
        
#         H1 = cv2.calcHist([query_im],[1],None,[256],[0,256])
#         # plt.plot(H1,color = 'b')
#         H2 = cv2.calcHist([cand_im],[1],None,[256],[0,256])
#         # plt.plot(H2,color = 'r')

#         #normalize hists
#         H1 = cv2.normalize(H1, H1, norm_type=cv2.NORM_L2)
#         H2 = cv2.normalize(H2, H2, norm_type=cv2.NORM_L2)

#         conf = cv2.compareHist(H1, H2, cv2.HISTCMP_INTERSECT)
#         # plt.title(conf)
#         # plt.xlim([0,256])
#         #plt.show()

#         if conf>best_cand_conf:
#             best_cand = cand
#             best_cand_conf = conf

#     # if cv2.waitKey(0) == ord('q'):
#     #     quit()

#     cv2.destroyAllWindows()
#     # print(f'Best match selected: {best_cand} with conf: {best_cand_conf}')
#     return best_cand, best_cand_conf



#Load individual camera trackings
num_cams = sum(1 for line in open(os.path.join(FRAME_NUM_PATH, f'S0{SEQ}.txt')))

dic_tracks_byframe = [utils.parse_aicity_rects(os.path.join(TRACK_PATH, cam,'gt', 'gt.txt')) for cam in CAM_NAMES]

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
    print(f'Matching camera {cam}')
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
                extension = 1*10 # how many seconds (by frame rate)
                fr_min_cand = fr_min_cand-extension if fr_min_cand>extension else 0
                fr_max_cand = fr_max_cand+extension if fr_max_cand+extension<max_frames[i] else max_frames[i]

                candidates=set()
                for f_cand in range(fr_min_cand, fr_max_cand+1):
                    if f'f_{f_cand}' in dic_tracks_byframe[i]:
                        for obj_cand in dic_tracks_byframe[i][f'f_{f_cand}']:
                            candidates.add(obj_cand['id'])

                match, conf = match_tracks(key_query, cam, candidates, i, dic_tracks, METHOD) if candidates else (-1, 1)
        
                for k_mt, el_mt in dic_tracks_byframe[i].items():
                    for obj_el in el_mt:
                        if obj_el['id'] == match and not(obj_el['mt_id'] != -1 and obj_el['mt_conf']>conf):
                            fr_aux = list(dic_tracks[cam][key_query].keys())[0]

                            for ei in dic_tracks_byframe[cam][f'f_{fr_aux}']:
                                if ei['id'] == key_query:
                                    obj_el['mt_id'] = ei['mt_id']
                            obj_el['mt_conf'] = conf

for i, det in enumerate(dic_tracks_byframe):
    os.makedirs(f'./mtrackings/S{str(SEQ).zfill(2)}/{str(CAM_NAMES[i]).zfill(3)}',exist_ok=True)
    utils.save_aicity_rects(f'./mtrackings/S{str(SEQ).zfill(2)}/{str(CAM_NAMES[i]).zfill(3)}/random.txt', det, True)