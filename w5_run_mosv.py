
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
from week5.utils import get_GT_path, get_VIDEO_path, get_DET_path


VIDEOS_LIST = ((1, list(range(1,6))), (3, list(range(10,16))), (4, list(range(16,41))))
DETECTIONS = ("yolo3", "ssd512", "mask_rcnn")

def run(sequence, camera, detection, save_video=False):
    #GT_PATH = get_GT_path(sequence, camera)
    DET_PATH = get_DET_path(sequence, camera, detection)
    VIDEO_PATH = get_VIDEO_path(sequence, camera)

    if not os.path.exists(VIDEO_PATH):
        print("Video does not exist.")
        return
    
    filename = f"{detection}_{args.tracker}_{args.tracker_life}_{args.threshold}"
    results_path = os.path.join(args.output, f"S{sequence:02d}", f"C{camera:03d}")
    results_path_videos = os.path.join("../videos", f"S{sequence:02d}", f"C{camera:03d}")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(results_path_videos):
        os.makedirs(results_path_videos)
    if save_video:
        writer = imageio.get_writer(os.path.join(results_path_videos, f"{filename}.mp4"), fps=25)
    reader = imageio.get_reader(VIDEO_PATH)
    tr_mgr = TracksManager(tracker_type = args.tracker, tracker_life = args.tracker_life, min_iou_th = 0.25)

    det_rects = utils.parse_aicity_rects(DET_PATH)
    
    trackid2colors = {}
    results_dict = {}
    for i, frame in tqdm(enumerate(reader)):
        if args.min != -1 and i < args.min:
            continue
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
            results_dict[frame_num].append(obj)

        if save_video:
            update_colors(detections, trackid2colors)
            frame = draw_tracking_bboxes(frame, detections, trackid2colors)
            writer.append_data(frame)

        if args.max != -1 and i >= args.max:
            break

    #print(f"DONE!")# {counter} frames processed")
    #print(f"Saving to... {args.output}")
    save_aicity_rects(os.path.join(results_path, f"{filename}.txt"), results_dict)
    if save_video:
        writer.close()
    reader.close()
    #print(f"Saved to '{results_path}'")


def main(args):
    for (seq, cameras) in VIDEOS_LIST:
        for camera in cameras:
            print(f"> S{seq:02d}-C{camera:03d}")
            for det in tqdm(DETECTIONS):
                run(seq, camera, det, save_video=args.video)
    
    




parser = argparse.ArgumentParser(description='Allows to run several tracking-by-detection algorithms.')
parser.add_argument('-o', '--output', type=str, default="../output", help="where results will be saved")
parser.add_argument('-d', '--detections', type=str, default=".", help="detections used for tracking. Options: {yolo3, ssd512, mask_rcnn}")
parser.add_argument('-t', '--tracker', type=str, default="identity", help='tracker used. Options: {"kalman", "kcf", "siamrpn_mobile", "siammask", "medianflow"}')
parser.add_argument('-th', '--threshold', type=float, default=0.5, help="threshold used to filter detections")
parser.add_argument('-tl', '--tracker_life', type=int, default=5, help="tracker life")
parser.add_argument('-v', '--video', action='store_true', help="tracker life")
parser.add_argument('-ov', '--only_video', action='store_true', help="tracker life")
parser.add_argument('-M', '--max', type=int, default=-1, help="max number of frames to run the tracker (by default it runs all video). Set to '-1' by default.")
parser.add_argument('-m', '--min', type=int, default=-1, help="min number of frames to run the tracker (by default it runs all video). Set to '-1' by default.")
args = parser.parse_args()

main(args)

# python w5_run_mosv.py --tracker kcf -tl 2
# /home/group01/anaconda3/envs/keras/bin/python w5_run_mosv.py --tracker siamrpn_mobile -tl 2