
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

    gif_buffer = []
    for i, frame in tqdm(enumerate(reader)):
        frame_key = f"f_{i}"
        if prvs is None:
            prvs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(frame)
            hsv[...,1] = 255
            continue

        detector_bboxes = []
        if frame_key in det_rects:
            detector_bboxes = [list(np.array(det["bbox"]).astype(int)) + [det["conf"], ] for det in det_rects[frame_key] if det["conf"] > args.threshold]
            #print(detector_bboxes)

        #detections = tr_mgr.update(frame, detector_bboxes)

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(hsv)
        hsv[...,1] = 255
        
        for (x0, y0, x, y, conf) in detector_bboxes:
            h, w = y - y0, x - x0
            h_m = int(0.1*h)
            w_m = int(0.1*w)
            x0, y0 = max(0, x0 - w_m), max(0, y0 - h_m)
            x, y = min(frame.shape[1], x + w_m), min(frame.shape[0], y + h_m)
            
            flow = cv2.calcOpticalFlowFarneback(prvs[y0:y, x0:x],frame[y0:y, x0:x], None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[y0:y, x0:x, 0] = ang*180/np.pi/2
            hsv[y0:y, x0:x, 2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)


        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        writer.append_data(rgb)
        gif_buffer.append(gif_preprocess(rgb))
        
        prvs = frame
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
parser.add_argument('-M', '--max', type=int, default=-1, help="max number of frames to run the tracker (by default it runs all video). Set to '-1' by default.")
args = parser.parse_args()

main(args)