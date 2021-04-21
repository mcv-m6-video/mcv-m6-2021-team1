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
dic_tracks = [utils.parse_aicity_rects(os.path.join(TRACK_PATH, f'c00{cam}\\gt\\gt.txt')) for cam in range(1, num_cams+1)]

print(dic_tracks[0]['f_1200'])

for id_n in dic_tracks[0]:
    