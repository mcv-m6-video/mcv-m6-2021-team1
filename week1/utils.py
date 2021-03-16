import random
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

from sklearn.metrics import average_precision_score
from matplotlib import pyplot as plt



def parse_aicity_rects(path):
    """
    Input:
        - Path to annotation xml in AI City format
    Output format
        dict[frame_num] = [[x1, y1, x2, y2]]
    """
    COL_NAMES = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']

    ret_dict = {}
    dtf = pd.read_csv(path, delimiter=',', names=COL_NAMES)

    for i, row in dtf.iterrows():
        frame_num = f'f_{int(row.frame) - 1}'

        if frame_num not in ret_dict:
            ret_dict[frame_num] = []
        obj = {
            'bbox': [row.bb_left, row.bb_top, row.bb_left+row.bb_width, row.bb_top+row.bb_height],
            'conf': float(row.conf)
        }
        ret_dict[frame_num].append(obj)

    return ret_dict


def parse_xml_rects(path):
    """
    Input:
        - Path to annotation xml in Pascal VOC format
    Output format:
        dict[frame_num] = [[x1, y1, x2, y2]]
    """
    tree = ET.parse(path)
    root = tree.getroot()
    frame_dict = {}
    for child in root:
        if child.tag=='track' and child.attrib['label']=='car':
            for x in child:
                d = x.attrib
                frame = f"f_{d['frame']}"
                if frame not in frame_dict:
                    frame_dict[frame] = []
                frame_dict[frame].append([float(d['xtl']), float(d['ytl']), 
                    float(d['xbr']), float(d['ybr'])])
    return frame_dict


def get_vertex_from_cwh(center, width, height):
    """
    Takes a 2d center, widht and height and return vertex in convenient format 
    """
    xtl = center[0] - width//2
    xbr = center[0] + width//2
    ytl = center[1] - height//2
    ybr = center[1] + height//2

    return xtl, ytl, xbr, ybr


def generate_noisy_bboxes(frame_dict, tol_dropout, std_pos, std_size, std_ar):
    """
    Input:
        - frame dict: gt_boxes
        - tol_droput: probability of removing boxes
        - std_pos: standartd deviation of gaussain controlling center position
        - std_size: standartd deviation of gaussain controlling area
        - std_ar: standartd deviation of gaussain controlling aspectr ratio
    """
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

                key = f'f_frame'
                if key not in noisy_dct:
                    noisy_dct[key] = {'bbox': [], 'conf':1.}
                noisy_dct[key]['bbox'].append(get_vertex_from_cwh(center, w, h))

    return noisy_dct


def get_rect_iou(a, b):
    """Return iou for a single a pair of boxes"""
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
    """Return iou for a frame"""
    list_iou = []

    for gt in gt_rects:
        max_iou = 0
        for obj in det_rects:
            det = obj['bbox']
            iou = get_rect_iou(det, gt)
            if iou > max_iou:
                max_iou = iou
        
        if max_iou != 0:
            list_iou.append(max_iou)

    return np.mean(list_iou)


def voc_ap(rec, prec):
    """ ap = voc_ap(rec, prec)
    Compute VOC AP given precision and recall.
    using the  VOC 07 11 point method.
    Based on https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
    (and the link I got from teams 3 or 4)
    """
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.
    return ap


def get_AP(gt_rects, det_rects, ovthresh=0.5):
    """
    gt_rects: ground truth rects in format dict[frame_num] = [[x1, y1, x2, y2]]
    det_rects: detection rects in format dict[frame_num] = [[x1, y1, x2, y2]]
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # parse gt_rects a esto
    # class_recs = {
    #     'imagename': {
    #         'bbox': [['x1', 'y1', 'x2', 'y2'], ['x1', 'y1', 'x2', 'y2']],
    #         'difficult':[True, False],
    #         'det': [False, False]
    #     }
    # }

    class_recs = {}
    npos = 0
    for frame, bboxs in gt_rects.items():
        class_recs[frame] = {
            'bbox': bboxs,
            'difficult': np.array([False]*len(bboxs)).astype(np.bool),
            'det': [False]*len(bboxs)
        }
        npos += len(bboxs)

    # image_ids = [0] # frame ids?
    # confidence = np.array([1]) # confidence for each detection
    # BB = np.array([['x1', 'y1', 'x2', 'y2']]) # for all detections

    image_ids = []
    confidence = []
    BB = []

    for frame, objs in det_rects.items():
        for obj in objs:
            image_ids.append(frame)
            confidence.append(obj['conf']) # unkwnown
            BB.append(obj['bbox'])

    confidence = np.array(confidence)
    BB = np.array(BB)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = np.array(R['bbox']).astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)

    # return rec, prec, ap
    return ap
