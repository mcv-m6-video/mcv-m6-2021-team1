import random
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

from sklearn.metrics import average_precision_score
from matplotlib import pyplot as plt


def parse_xml_rects(path):
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


def get_vertex_from_cwh(center, width, height):
    xtl = center[0] - width//2
    xbr = center[0] + width//2
    ytl = center[1] - height//2
    ybr = center[1] + height//2

    return xtl, ytl, xbr, ybr


def generate_noisy_bboxes(frame_dict, tol_dropout, std_pos, std_size, std_ar):

    noisy_dct = {}

    for frame, bbs in frame_dict.items():
        for bb in bbs:
            
            if random.random() > tol_dropout:     #random dropout
                
                xtl, ytl, xbr, ybr = bb

                w = abs(xtl - xbr)
                h = abs(ytl - ybr)

                #position noise
                center = np.array([w/2+xtl, h/2+ytl])
                center += np.random.normal(0, std_pos*w, 2)
                
                #size noise
                scale_f = np.random.normal(1, std_size)
                h *= scale_f
                w *= scale_f

                #Aspect ratio noise
                h *= np.random.normal(1, std_ar)
                w *= np.random.normal(1, std_ar)


                if frame not in noisy_dct:
                    noisy_dct[frame] = []
                noisy_dct[frame].append(get_vertex_from_cwh(center, w, h))

    return noisy_dct


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


# def parse_aicity_rects(path):
# Not sure why this function is twice
#     ret_dict = {}
#     dtf = pd.read_csv(path, delimiter=',', names=COL_NAMES)

#     for i, row in dtf.iterrows():
#         if row.frame not in ret_dict:
#             ret_dict[row.frame] = []
#         ret_dict[row.frame].append([row.bb_left, row.bb_top, row.bb_left+row.bb_width, row.bb_top+row.bb_height])

#     return ret_dict

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


def get_AP(gt_rects, det_rects):

    correct = np.zeros(len(det_rects))
    conf = np.ones(len(det_rects)) # simulation of confidence levels

    previous_detections = set()
    for i, det in enumerate(det_rects):
        for j, gt in enumerate(gt_rects):
            iou = get_rect_iou(det, gt)

            if iou > 0.5 and j not in previous_detections:
                correct[i] = 1
                previous_detections.add(j)

    # print('Scikit learn AP', average_precision_score(correct, conf, 'samples'))

    ac_correct = np.array([np.sum(correct[:i]) for i in range(1, len(det_rects))])

    precision = ac_correct / (np.array(range(len(ac_correct)))+1)
    recall = ac_correct / len(gt_rects)

    # Values at infinity
    precision = np.hstack([precision, 0])
    recall = np.hstack([recall, 1])

    # Interpolate
    interp_precision = []
    sample_idx = np.linspace(0, 1, 11)
 
    sample_values = []
    for i in sample_idx:
        cont = -1
        for v in recall:
            cont+=1
            if v > i: 
                break
        sample_values.append(precision[cont])

    return np.mean(sample_values)


def rectangle_mot_format(img, left, top, width, height, color, thickness=None, conf=1, id=1):
    left = int(left)
    top = int(top)
    width = int(width)
    height = int(height)

    return cv2.rectangle(img, (left, top), (left+width, top+height), color, thickness)
  

def parse_aicity_rects(path):
    COL_NAMES = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']

    ret_dict = {}
    dtf = pd.read_csv(path, delimiter=',', names=COL_NAMES)

    for i, row in dtf.iterrows():
        if row.frame not in ret_dict:
            ret_dict[row.frame] = []
        ret_dict[row.frame].append([row.bb_left, row.bb_top, row.bb_left+row.bb_width, row.bb_top+row.bb_height])

    return ret_dict