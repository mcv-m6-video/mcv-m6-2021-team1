
import numpy as np
import cv2
from random import randint
import pandas as pd

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


def compute_intersection_over_union(boxA, boxB):
    # Format: XYXY (top-left and bottom-right)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def draw_tracking_bboxes(frame, dets, trackid2colors):
    # trackid2colors: dictionary with key as track id and value as color in format (RED, GREEN, BLUE) (int values)
    # paint bounding boxes
    for det in dets:
        color = trackid2colors[det.track_id]
        bbox = det.bbox  # detection
        (x0, y0, x, y) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        if det.from_tracker:
            cv2.rectangle(frame, (x0, y0), (x, y), color, 2)
        else:
            cv2.rectangle(frame, (x0, y0), (x, y), (0,0,0), 4)
        if det.from_tracker and det.track_id != -1:
            cv2.putText(frame, str(det.track_id), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame


def update_colors(detections, trackid2colors):
    for det in detections:
        if det.track_id not in trackid2colors:
            # we set a color for this track
            trackid2colors[det.track_id] = (randint(0, 255), randint(0, 255), randint(0, 255))
