
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import imageio
import argparse
from tqdm import tqdm
import sys
from week2 import utils
from week3.kalman.tracker import TracksManager
from week3.kalman.tracking_utils import draw_tracking_bboxes, update_colors, save_aicity_rects

# 2141 frames in total
TOTAL_FRAMES = 2141

VIDEO_PATH = "../data/AICity_data/train/S03/c010/vdo.avi"
GT_RECTS_PATH = "../data/ai_challenge_s03_c010-full_annotation.xml"
AI_GT_RECTS_PATH = "../data/AICity_data/train/S03/c010/gt/gt.txt"
DETECTIONS = {"retinanetpre": "week3/detections/m6-aicity_retinanet_R_50_FPN_3x_rp128.txt", 
                "maskrcnnpre": "week3/detections/det_mask_rcnn.txt",
                "ssdpre": "week3/detections/det_ssh512.txt",
                "yolopre": "week3/detections/det_yolo3.txt",
                #"retinanet101pre": "",
                }

def main(args):
    if args.detections not in DETECTIONS:
        raise Exception("Detections not supported")
    AI_GT_RECTS_PATH = DETECTIONS[args.detections]
    DET_NAME = args.detections

    if not os.path.exists(VIDEO_PATH):
        print("Video does not exist.")
        return

    filename = f"{args.tracker}_{args.tracker_life}_{args.threshold}"
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
    for i, frame in tqdm(enumerate(reader)):
        frame_key = f"f_{i}"
        detector_bboxes = []
        if frame_key in det_rects:
            detector_bboxes = [list(np.array(det["bbox"]).astype(int)) + [det["conf"], ] for det in det_rects[frame_key] if det["conf"] > args.threshold]

        detections = tr_mgr.update(frame, detector_bboxes)
        for det in detections:
            if det.from_tracker:
                continue
            
            frame_num = f'f_{int(i+1)}' # it has to start at frame 1 instead of 0
            if frame_num not in results_dict:
                results_dict[frame_num] = []
            b = det.bbox
            obj = {
                'bbox': [int(b[0]), int(b[1]), int(b[2]), int(b[3])],
                'conf': float(det.score),
                'id': int(det.track_id)
            }
            #print(obj)
            results_dict[frame_num].append(obj)

        #print(detector_bboxes)
        #print(detections)
        #print("______________")
        #print(detections)
        update_colors(detections, trackid2colors)
        #frame = draw_tracking_bboxes(frame, detections, trackid2colors)
        #writer.append_data(frame)

        if args.max != -1 and counter >= args.max:
            break

    print(f"DONE!")# {counter} frames processed")
    print(f"Saving to... {args.output}")
    save_aicity_rects(os.path.join(args.output, f"{filename}.txt"), results_dict)
    writer.close()
    reader.close()
    print(f"Saved to '{results_path}'")




parser = argparse.ArgumentParser(description='Extract foreground from video.')
parser.add_argument('-o', '--output', type=str, default=".", help="where results will be saved")
parser.add_argument('-d', '--detections', type=str, default=".", help="detections used")
parser.add_argument('-t', '--tracker', type=str, default="identity", help="tracker used")
parser.add_argument('-th', '--threshold', type=float, default=0.5, help="threshold for detections")
parser.add_argument('-tl', '--tracker_life', type=int, default=5, help="tracker life")
parser.add_argument('-M', '--max', type=int, default=-1, help="max number of frames for which to extract foreground. Set to '-1' by default.")
#parser.add_argument('-s', '--save', type=bool, store_=-1, help="max number of frames for which to extract foreground. Set to '-1' by default.")
args = parser.parse_args()

main(args)