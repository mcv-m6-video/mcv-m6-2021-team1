""" 
    MOTChallenge format [frame, ID, left, top, width, height, 1, -1, -1, -1].
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z> 
"""
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


from matplotlib import pyplot as plt

COL_NAMES = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']


def parse_aicity_rects(path):
    ret_dict = {}
    dtf = pd.read_csv(path, delimiter=',', names=COL_NAMES)

    for i, row in dtf.iterrows():
        if row.frame not in ret_dict:
            ret_dict[row.frame] = []
        ret_dict[row.frame].append([row.bb_left, row.bb_top, row.bb_left+row.bb_width, row.bb_top+row.bb_height])

    return ret_dict

def parse_xml_rects(path):
    # './Data/ai_challenge_s03_c010-full_annotation.xml'
    tree = ET.parse(path)

    root = tree.getroot()
    frame_dict = {}
    for child in root:
        if child.tag=='track' and child.attrib['label']=='car':
            for x in child:
                d = x.attrib
                if int(d['frame']) not in frame_dict:
                    frame_dict[int(d['frame'])] = []
                frame_dict[int(d['frame'])].append([float(d['xtl']), float(d['ytl']), 
                    float(d['xbr']), float(d['ybr'])])
    return frame_dict

def get_rect_iou(a, b):
    x11, y11, x12, y12 = a
    x21, y21, x22, y22 = b

    xA = max(x11,x21)
    yA = max(y11,y21)
    xB = min(x12,x22)
    yB = min(y12,y22)
     
    # respective area of ​​the two boxes
    boxAArea=(x12-x11)*(y12-y11)
    boxBArea=(x22-x21)*(y22-y21)
     
     # overlap area
    interArea=max(xB-xA,0)*max(yB-yA,0)
     
     # IOU
    return interArea/(boxAArea+boxBArea-interArea)

def get_frame_iou(gt_rects, det_rects):
    list_iou = []

    for gt in gt_rects:
        max_iou = 0
        for det in det_rects:
            iou = get_rect_iou(det, gt)
            if iou > max_iou:
                max_iou = iou
        list_iou.append(max_iou)
    return np.mean(list_iou)

VIDEO_PATH = '../../data/AICity_data/train/S03/c010/vdo.avi'
RCNN_PATH = '../../data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt'
GT_PATH = '../../data/ai_challenge_s03_c010-full_annotation.xml'

det_all_rects = parse_aicity_rects(RCNN_PATH)
gt_all_rects = parse_xml_rects(GT_PATH)

# rects = []
# rects.append({'name':'gt', 'data': pd.read_csv(GT_PATH, delimiter=',', names=COL_NAMES), 'color':(0, 255,0)})
# rects.append({'name':'mask-rcnn', 'data': pd.read_csv(DET_PATH, delimiter=',', names=COL_NAMES), 'color':(0,0,255)})

# Render video
cap = cv2.VideoCapture(VIDEO_PATH)

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
        miou = get_frame_iou(gt_rects, det_rects)
        print('Frame', frame_cont, 'iou:', miou)

    cv2.imshow('frame',cv2.resize(frame, tuple(np.int0(0.5*np.array(frame.shape[:2][::-1])))))

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()
    frame_cont += 1

cap.release()
cv2.destroyAllWindows()