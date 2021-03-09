""" 
    MOTChallenge format [frame, ID, left, top, width, height, 1, -1, -1, -1].
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z> 
"""
import cv2
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

def rectangle_mot_format(img, left, top, width, height, color, thickness=None, conf=1, id=1):
    left = int(left)
    top = int(top)
    width = int(width)
    height = int(height)

    return cv2.rectangle(img, (left, top), (left+width, top+height), color, thickness)

VIDEO_PATH = '../../data/AICity_data/train/S03/c010/vdo.avi'
GT_PATH = '../../data/AICity_data/train/S03/c010/gt/gt.txt'
DET_PATH = '../../data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt'

COL_NAMES = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']

rects = []
rects.append({'name':'gt', 'data': pd.read_csv(GT_PATH, delimiter=',', names=COL_NAMES), 'color':(0, 255,0)})
rects.append({'name':'mask-rcnn', 'data': pd.read_csv(DET_PATH, delimiter=',', names=COL_NAMES), 'color':(0,0,255)})

# Render video
cap = cv2.VideoCapture(VIDEO_PATH)

frame_cont = 0
ret, frame = cap.read()
while(ret):

    for r in rects:
        try:
            data = r['data']
            for idx in data.frame[data.frame == frame_cont].index:
                d = data.iloc[idx]
                frame = rectangle_mot_format(frame, d.bb_left, d.bb_top, d.bb_width, d.bb_height, r['color'], 2)
        except IndexError:
            print(f'No {r["name"]} data for frame', frame_cont)

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()
    frame_cont += 1

cap.release()
cv2.destroyAllWindows()