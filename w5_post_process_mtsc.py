
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
from week3.kalman.tracking_utils import draw_tracking_bboxes, update_colors
from week3.utils import parse_aicity_rects, save_aicity_rects
from week5.utils import get_GT_path, get_VIDEO_path, get_DET_path
import seaborn as sns
import matplotlib.pyplot as plt

TRAINING_VIDEOS = ((1, list(range(1,6))), (3, list(range(10,16))))
VIDEOS_LIST = ((1, list(range(1,6))), (3, list(range(10,16))), (4, list(range(16,41))))
VIDEOS_LIST_TR = ((4, list(range(16,41))), )
DETECTIONS = ("yolo3", "ssd512", "mask_rcnn")
COLORS = ("orange", "green", "blue")


def gt_sizes_analysis(args):
    areas = []
    for (sequence, cameras) in TRAINING_VIDEOS:
        for camera in cameras:
            print(f"> S{sequence:02d}-C{camera:03d}")
            GT_PATH = get_GT_path(sequence, camera)
                    
            # dict[frame_num] = [{'bbox': [x1, y1, x2, y2]. 'conf':x, 'id':i}]
            det_rects = parse_aicity_rects(GT_PATH)

            tracklets = {} # tracklets["id"]=[[cx, cy, w, h], ... [x, y, w, h]]
            for frame_num in det_rects:
                for detection in det_rects[frame_num]:
                    bbox = detection["bbox"]
                    tr_id = detection["id"]
                    if tr_id not in tracklets:
                        tracklets[tr_id] = []

                    areas.append(round(bbox[2]-bbox[0]) * round(bbox[3]-bbox[1])) # width * height

    print(f"Minimum area: {np.min(areas)}")
    plt.figure()
    plt.title(f"Area distribution of GT detections")
    plt.xlabel("Area")
    sns.distplot(areas, bins=1000, hist=True, kde=False, norm_hist=True)
    plt.xlim((0, 100000))
    plt.savefig(f"plot_areas.png", dpi=250)
    


def parked_cars_analysis(args):
    stds = []
    stds_per_detector = {}
    for det in DETECTIONS:
        stds_per_detector[det] = []

    for (sequence, cameras) in TRAINING_VIDEOS:
        for camera in cameras:
            print(f"> S{sequence:02d}-C{camera:03d}")
            results_path = os.path.join(args.input, f"S{sequence:02d}", f"C{camera:03d}")
            for root, dirs, files in os.walk(results_path, topdown=False):
                for name in files:
                    from_path = os.path.join(root, name)
                    to_folder = from_path.replace(root, args.output)
                    to_path = os.path.join(to_folder, name)
                    #print(from_path)
                    #print(to_path)
                    #we find the type of detection used
                    detector = None
                    for det in DETECTIONS:
                        if det in from_path:
                            detector = det
                    if detector is None:
                        raise Exception("Tracking results with no detector associated.")
                    
                    if not os.path.exists(to_folder):
                        os.makedirs(to_folder)
                    
                    # dict[frame_num] = [{'bbox': [x1, y1, x2, y2]. 'conf':x, 'id':i}]
                    det_rects = parse_aicity_rects(from_path)

                    tracklets = {} # tracklets["id"]=[[cx, cy, w, h], ... [x, y, w, h]]
                    for frame_num in det_rects:
                        for detection in det_rects[frame_num]:
                            bbox = detection["bbox"]
                            tr_id = detection["id"]
                            if tr_id not in tracklets:
                                tracklets[tr_id] = []

                            tracklets[tr_id].append([round((bbox[0] + bbox[2])/2), round((bbox[1]+bbox[3])/2), round(bbox[2]-bbox[0]), round(bbox[3]-bbox[1])])
                    
                    tracklets_info = {}
                    stds = []
                    for tr_id in tracklets:
                        tr_mean = np.round(np.mean(tracklets[tr_id], axis=0)).astype(int)
                        tr_std = np.round(np.std(tracklets[tr_id], axis=0)).astype(int)
                        std_norm = int(round(np.linalg.norm(tr_std)))
                        #print(std_norm)
                        stds.append(std_norm)
                    stds_per_detector[detector] += stds

    for det, col in zip(DETECTIONS, COLORS):
        plt.figure()
        plt.title(f"Detector: {det}")
        plt.xlabel("STD norm of (cx, cy, w, h)")
        sns.distplot(stds_per_detector[det], bins=100, color=col, hist=True, kde=False, norm_hist=True)
        plt.xlim((0, 300))
        plt.ylim(0, 0.04)
        plt.savefig(f"plot_{det}.png", dpi=250)
                    #save_aicity_rects(to_path, results_dict)


def run(args, norm_th=25, area_th=8874):
    for (sequence, cameras) in VIDEOS_LIST:
        for camera in cameras:
            print(f"> S{sequence:02d}-C{camera:03d}")
            results_path = os.path.join(args.input, f"S{sequence:02d}", f"C{camera:03d}")

            for root, dirs, files in os.walk(results_path, topdown=False):
                for name in files:
                    to_remove_ids = []

                    from_path = os.path.join(root, name)
                    to_folder = root.replace(args.input, args.output)
                    to_path = os.path.join(to_folder, name)
                    #print(from_path)
                    print(to_path)
                    
                    if not os.path.exists(to_folder):
                        os.makedirs(to_folder)
                    
                    # dict[frame_num] = [{'bbox': [x1, y1, x2, y2]. 'conf':x, 'id':i}]
                    det_rects = parse_aicity_rects(from_path)

                    tracklets = {} # tracklets["id"]=[[cx, cy, w, h], ... [x, y, w, h]]
                    for frame_num in det_rects:
                        for detection in det_rects[frame_num]:
                            bbox = detection["bbox"]
                            tr_id = detection["id"]
                            if tr_id not in tracklets:
                                tracklets[tr_id] = []

                            tracklets[tr_id].append([round((bbox[0] + bbox[2])/2), round((bbox[1]+bbox[3])/2), round(bbox[2]-bbox[0]), round(bbox[3]-bbox[1])])
                    
                    to_remove_ids = [tr_id for tr_id in tracklets if np.linalg.norm(np.round(np.std(tracklets[tr_id], axis=0)).astype(int)) < norm_th]
                    
                    cleaned_dict = {}
                    for frame_num in det_rects:
                        cleaned_dict[frame_num] = [det for det in det_rects[frame_num] if det["id"] not in to_remove_ids and (det["bbox"][2]-det["bbox"][0])*(det["bbox"][3]-det["bbox"][1]) >= area_th]

                    save_aicity_rects(to_path, cleaned_dict)
            


def main(args):
    #parked_cars_analysis(args)
    #gt_sizes_analysis(args)
    run(args, norm_th = args.th)

    


parser = argparse.ArgumentParser(description='Post-process detections')
parser.add_argument('-o', '--output', type=str, default="../output_post", help="where post-processed results will be saved")
parser.add_argument('-i', '--input', type=str, default="../output", help="where the detections will be loaded from")
parser.add_argument('-th', '--th', type=int, default=25, help="where the detections will be loaded from")
args = parser.parse_args()

main(args)


