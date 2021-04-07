
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import imageio
import argparse
from tqdm import tqdm
import sys
from week3 import utils
from week3.kalman.tracker import TracksManager
from week3.kalman.tracking_utils import draw_tracking_bboxes, update_colors, save_aicity_rects


# 2141 frames in total
TOTAL_FRAMES = 2141

VIDEO_PATH = "../data/AICity_data/train/S03/c010/vdo.avi"
AI_GT_RECTS_PATH = "../data/AICity_data/train/S03/c010/gt/gt.txt"

DETECTIONS = {"retinanetpre": "week3/detections/det_retina50.txt", 
                "maskrcnnpre": "week3/detections/det_mask_rcnn.txt",
                "ssdpre": "week3/detections/det_ssd512.txt",
                "yolopre": "week3/detections/det_yolo3.txt",
                "retinanet101pre": "week3/detections/det_retina101.txt",
                }




def gif_preprocess(im, width=512):
    im = utils.resize_keep_ap(im, width=width)
    return im

def main(args):
    if args.detections not in DETECTIONS:
        raise Exception("Detections not supported")
    AI_GT_RECTS_PATH = DETECTIONS[args.detections]
    if not os.path.exists(VIDEO_PATH):
        print("Video does not exist.")
        return

    filename = f"{args.detections}_{args.tracker}_{args.tracker_life}_{args.threshold}"
    results_path = args.output
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    writer = imageio.get_writer(f"{filename}.mp4", fps=25)
    reader = imageio.get_reader(VIDEO_PATH)
    tr_mgr = TracksManager(tracker_type = args.tracker, tracker_life = args.tracker_life, min_iou_th = 0.5)

    det_rects = utils.parse_aicity_rects(AI_GT_RECTS_PATH)
    
    #gt_rects_detformat = {f: [{'bbox': r, 'conf':1} for r in v] for f, v in gt_rects.items()}
    trackid2colors = {}
    results_dict = {}
    prvs = None

    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )
    lk_params = dict( winSize = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
 
    color = np.random.randint(0,255,(100,3))
    gif_buffer = []
    for i, frame in tqdm(enumerate(reader)):
        if args.min != -1 and i < args.min:
            continue
            
        frame_key = f"f_{i}"
        if prvs is None:
            prvs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(prvs, mask = None, **feature_params)
            mask = np.zeros_like(frame) 
            continue

        detector_bboxes = []
        if frame_key in det_rects:
            detector_bboxes = [list(np.array(det["bbox"]).astype(int)) + [det["conf"], ] for det in det_rects[frame_key] if det["conf"] > args.threshold]
            #print(detector_bboxes)

        #detections = tr_mgr.update(frame, detector_bboxes)

        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(prvs, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        # dibuja las lineas
        for j,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[j].tolist(), 5)
            frame = cv2.circle(frame,(a,b),10,color[j].tolist(),-1)

        img = cv2.add(frame,mask)
        writer.append_data(img)
        gif_buffer.append(gif_preprocess(img))
        
        prvs = frame_gray
        p0 = good_new.reshape(-1,1,2)
        if args.max != -1 and i >= args.max:
            break

    path = f"{filename}.gif"
    imageio.mimsave(path, gif_buffer)
    #optimize(path)
    print(f"DONE!")# {counter} frames processed")
    print(f"Saving to... {args.output}")
    writer.close()
    reader.close()
    print(f"Saved to '{results_path}'")




parser = argparse.ArgumentParser(description='Allows to run several tracking-by-detection algorithms.')
parser.add_argument('-o', '--output', type=str, default=".", help="where results will be saved")
parser.add_argument('-d', '--detections', type=str, default="retinanet101pre", help="detections used for tracking. Options: {retinanetpre, retinanet101pre, maskrcnnpre, ssdpre, yolopre}")
parser.add_argument('-t', '--tracker', type=str, default="iou", help='tracker used. Options: {"kalman", "kcf", "siamrpn_mobile", "siammask", "medianflow"}')
parser.add_argument('-th', '--threshold', type=float, default=0.5, help="threshold used to filter detections")
parser.add_argument('-tl', '--tracker_life', type=int, default=5, help="tracker life")
parser.add_argument('-m', '--min', type=int, default=-1, help="min number of frames to run the tracker (by default it runs all video). Set to '-1' by default.")
parser.add_argument('-M', '--max', type=int, default=-1, help="max number of frames to run the tracker (by default it runs all video). Set to '-1' by default.")
args = parser.parse_args()

main(args)