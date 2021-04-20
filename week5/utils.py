import os
import random
import cv2
import imageio
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.metrics import average_precision_score
from matplotlib import pyplot as plt
import colorsys


color_id =  {}

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return pick #boxes[pick].astype("int")

def get_random_col():
    h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
    r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
    return (b, g, r)


def fix_zero_idx(path):

    dt_rec = parse_aicity_rects(path, 0)
    dt_fix = {}
    for f in dt_rec:
        dt_fix[f'f_{int(f[2:])+1}'] = dt_rec[f]
    save_aicity_rects(path,dt_fix)
    return


def parse_aicity_rects(path, zero_index=1):
    """
    Input:
        - Path to annotation xml in AI City format
    Output format
        dict[frame_num] = [{'bbox': [x1, y1, x2, y2]. 'conf':x, 'id':i}]
    """
    COL_NAMES = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']

    ret_dict = {}
    dtf = pd.read_csv(path, delimiter=',', names=COL_NAMES)

    for i, row in dtf.iterrows():
        frame_num = f'f_{int(row.frame) - zero_index}'

        if frame_num not in ret_dict:
            ret_dict[frame_num] = []

        obj = {
            'bbox': [row.bb_left, row.bb_top, row.bb_left+row.bb_width, row.bb_top+row.bb_height],
            'conf': float(row.conf),
            'id': int(row.id)
        }
        ret_dict[frame_num].append(obj)

    return ret_dict

def save_aicity_rects(path, det_rects):

    COL_NAMES = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']
    dic_csv = {
        'frame' : [], 'id': [], 'bb_left': [], 'bb_top': [], 'bb_width': [], 'bb_height': [], 'conf': [], 'x': [], 'y': [], 'z': []
    }
    for f in det_rects:
        for det in det_rects[f]:
            dic_csv['frame'].append(f[2:])
            dic_csv['id'].append(det['id'])
            dic_csv['bb_left'].append(det['bbox'][0])
            dic_csv['bb_top'].append(det['bbox'][1])
            dic_csv['bb_width'].append(det['bbox'][2]-det['bbox'][0])
            dic_csv['bb_height'].append(det['bbox'][3]-det['bbox'][1])
            dic_csv['conf'].append(det['conf'])
            dic_csv['x'].append(-1)
            dic_csv['y'].append(-1)
            dic_csv['z'].append(-1)

    df = pd.DataFrame(dic_csv, columns=COL_NAMES)
    df.to_csv(path, index = False, header=False)



def parse_xml_rects(path, remove_static=False):
    """
    Input:
        - Path to annotation xml in Pascal VOC format
    Output format:
        dict[frame_num] = [{'bbox':[x1, y1, x2, y2], 'conf': 1, 'id': -1}]
    """
    tree = ET.parse(path)
    root = tree.getroot()
    frame_dict = {}
    for child in root:
        if child.tag=='track' and child.attrib['label']=='car':
            track_id = int(child.attrib['id'])
            for x in child:
                d = x.attrib
                frame = f"f_{d['frame']}"

                if x[0].text == 'true' and remove_static: #
                    if frame not in frame_dict:
                        frame_dict[frame] = []
                    continue

                if frame not in frame_dict:
                    frame_dict[frame] = []
                frame_dict[frame].append(
                    {
                    'conf': 1,
                    'bbox': [float(d['xtl']), float(d['ytl']), 
                    float(d['xbr']), float(d['ybr'])],
                    'id': track_id
                    })
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


def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/10
    return 1


def pretty_rects(im, objs, name, color, conf_thresh=0.0, tracking = False):
    for obj in objs:
        if float(obj["conf"]) < conf_thresh:
            continue
        if tracking:
            if str(obj['id']) not in color_id:
                color_id[str(obj['id'])] = get_random_col()
            color = color_id[str(obj['id'])]

        bb = obj['bbox']
        h = bb[3] - bb[1]
        w = bb[2] - bb[0]
        # Paint box
        im = cv2.rectangle(im, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), color, 2)

        # Write name and conf
        text = f'{name} {obj["id"]}- {int(100*obj["conf"])} %'
        # "Background" for label
        im = cv2.rectangle(im, 
            (int(bb[0]), int(bb[1])),
            (int(bb[0]+0.7*w), int(bb[1]+0.2*h)),
            color, -1)
        # Print name, id and conf
        cv2.putText(im, text, (int(bb[0]), int(bb[1]+0.15*h)), cv2.FONT_HERSHEY_COMPLEX_SMALL, get_optimal_font_scale(text, 0.7*w), (0,0,0))
    return im


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


def imshow_rects(im, rect_list, name='result', disp=False):
    #im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for det in rect_list:
        rects = det['rects']
        color = det['color']

        for obj in rects:
            r = obj['bbox']
            im = cv2.rectangle(im, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), color, 3)
    if disp:
        display_resized(name, im)
    return im


def resize_keep_ap(im, sf=None, width=None, height=None):
    shape = im.shape[:2][::-1]

    if sf:
     return cv2.resize(im, tuple(np.int0(sf*np.array(shape))))

    if width:
        return cv2.resize(im, (width, width*shape[1]//shape[0]))

    if height:
        return cv2.resize(im, (height*shape[0]//shape[1], height))

    return im


def display_resized(name, im, sf=0.5):
    im = cv2.resize(im, tuple(np.int0(sf*np.array(im.shape[:2][::-1]))))
    cv2.imshow(name, im)
    k = cv2.waitKey(5)
    if k == ord('q'):
        quit()


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
    for frame, objs in gt_rects.items():
        class_recs[frame] = {
            'bbox': [obj['bbox'] for obj in objs],
            'difficult': np.array([False]*len(objs)).astype(np.bool),
            'det': [False]*len(objs)
        }
        npos += len(objs)

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

