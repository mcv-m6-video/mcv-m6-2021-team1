""" 
    MOTChallenge format [frame, ID, left, top, width, height, 1, -1, -1, -1].
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z> 
"""
import os
import cv2
import numpy as np
import pandas as pd
import utils

from matplotlib.pyplot import Figure
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

#PATHS
RUN_NAME = 'mask_rcnn'
annotations_file = '../../data/ai_challenge_s03_c010-full_annotation.xml'
VIDEO_PATH = '../../data/AICity_data/train/S03/c010/vdo.avi'
GT_PATH = '../../data/AICity_data/train/S03/c010/gt/gt.txt'
DET_PATH = '../../data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt'
DET_PATH = '../../data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt'
DET_PATH = '../../data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt'


def main():

    if not os.path.exists(f'runs/{RUN_NAME}/'):
        os.mkdir(f'runs/{RUN_NAME}/')

    gt_all_rects = utils.parse_xml_rects(annotations_file)

    tol_dropout = 0.1
    std_pos = 0.2
    std_size = 0.5
    std_ar = 0.3
    det_all_rects = utils.generate_noisy_bboxes(gt_all_rects, tol_dropout, std_pos, std_size, std_ar)

    # Render video
    cap = cv2.VideoCapture(VIDEO_PATH)
    miou_over_time = []
    map_over_time = []
    frame_cont = 0
    ret, frame = cap.read()

    while(ret):
    
        gt_rects = gt_all_rects.get(frame_cont, None)
        det_rects = det_all_rects.get(frame_cont, None)

        if gt_rects:
            for r in gt_rects:
                frame = cv2.rectangle(frame, (int(r[0]), int(r[1])),  (int(r[2]), int(r[3])), (0, 255, 0))

        if det_rects:
            for r in det_rects:
                frame = cv2.rectangle(frame, (int(r[0]), int(r[1])),  (int(r[2]), int(r[3])), (0, 0, 255))

        if gt_rects and det_rects:
            miou = utils.get_frame_iou(gt_rects, det_rects)
            miou_over_time.append(miou)
            ap = utils.get_AP(gt_rects, det_rects)
            map_over_time.append(ap)
            print('Frame', frame_cont, 'iou:', miou, 'ap:', ap)
        else:
            miou_over_time.append(miou)
        
        plt.plot(miou_over_time)
        plt.xlim([0, 2140])
        display_frame = cv2.resize(frame, tuple(np.int0(0.5*np.array(frame.shape[:2][::-1]))))
        cv2.imshow('frame',display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        plt.savefig(f'runs/{RUN_NAME}/iou_plt_{frame_cont}.png')
        cv2.imread(f'runs/{RUN_NAME}/frame_{frame_cont}.png')

        ret, frame = cap.read()
        frame_cont += 1

    cap.release()
    cv2.destroyAllWindows()

main()