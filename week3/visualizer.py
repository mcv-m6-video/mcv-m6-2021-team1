"""
Script for visualizing results for different models, generating gifs, etc. We are splitting
detection and visualization, so inputs to this script are simply json/pkl (to define) files with
detections in the agreed format.

Input data:
        - video?
        - detector name (str)
        - color ([int, int, int])
        - Detections: 
            Dictionary with key 'f_{num_frame}' and value a list of "obj" with
            the followong format:

                obj = {
                    'id': {tracking id},
                    'conf': {condifence level},
                    'bbox': [x1, y1, x2, y2]
                }
"""
import os
import random
import colorsys
import datetime as dt
import cv2
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt

import utils
import detection

TOTAL_FRAMES = 2141
VIDEO_PATH = "../../data/AICity_data/train/S03/c010/vdo.avi"
GT_RECTS_PATH = "../../data/ai_challenge_s03_c010-full_annotation.xml"
AI_GT_RECTS_PATH = "../../data/AICity_data/train/S03/c010/gt/gt.txt"
OUT_DIR = 'out_visualizer'

AP_thresh = 0.5
conf_thresh = 0.5

def gif_preprocess(im, width=512):
    im = utils.resize_keep_ap(im, width=width)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def get_random_col():
    h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
    r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
    return (b, g, r)

# Load detections
print('Loading detections...')
detections = [
    {
        'name': 'gt',
        'full-name': 'Ground truth',
        'color': (0, 255, 0),
        'rects': utils.parse_xml_rects(GT_RECTS_PATH)
    },
    # {
    #     'name': 'aigt',
    #     'full-name': 'AI City GT',
    #     'color': (0, 0, 255),
    #     'rects': utils.parse_aicity_rects(AI_GT_RECTS_PATH)
    # },
    {
        'name': 'retina 50',
        'full-name': 'Retina Net R50 FPN 3x rp 128',
        'color': get_random_col(),
        'rects': utils.parse_aicity_rects('./detections/m6-aicity_retinanet_R_50_FPN_3x_rp128.txt', zero_index=0)
    },
    {
        'name': 'yolo',
        'full-name': 'YOLO',
        'color': get_random_col(),
        'rects': utils.parse_aicity_rects('./detections/det_yolo3.txt', zero_index=1)
    },
        {
        'name': 'ssd',
        'full-name': 'single shot detection 512',
        'color': get_random_col(),
        'rects': utils.parse_aicity_rects('./detections/det_ssd512.txt', zero_index=1)
    },
        {
        'name': 'rcnn',
        'full-name': 'Mask RCNN',
        'color': get_random_col(),
        'rects': utils.parse_aicity_rects('./detections/det_mask_rcnn.txt', zero_index=1)
    },

]

def main(display=True):

    # Create output dirs
    print('Create output dir...')
    now = dt.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    out_dir = os.path.join(OUT_DIR, now)

    os.makedirs(os.path.join(out_dir, 'snapshots'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'gifs'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'frames'), exist_ok=True)

    # Render video
    print('Start visualization...')
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_cont = 0
    ret, frame = cap.read()
    
    WAIT_TIME_LIST = [30, 60, 0, 10] # Available display speeds
    wait_time_idx = 0 # start at 33 FPS (15 ms per frame)

    gif_buffer = None
    gifs_to_save = {}
    while(ret):
        # Render detections
        for det in detections:
            frame = utils.pretty_rects(frame, det['rects'].get(f'f_{frame_cont}', []), det['name'], det['color'], conf_thresh=conf_thresh)
 
        if display:

            # Display info
            wait_time = WAIT_TIME_LIST[wait_time_idx % len(WAIT_TIME_LIST)]
            FPS = int(1e3/wait_time if wait_time != 0 else 0)
            rec = '[REC]' if gif_buffer else '[___]'

            h, w = frame.shape[:2]
            x, y = int(w*0.75), int(h*0.8)

            info = f'{FPS} FPS {rec}'
            h, w = frame.shape[:2]
            frame = cv2.putText(frame, info, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                utils.get_optimal_font_scale(info, w*0.1), (0, 0, 255), 2)

            for det in detections:
                y += 25
                frame = cv2.putText(frame, det['full-name'], (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                    utils.get_optimal_font_scale(info, w*0.1), det['color'], 2)

            # Display
            display_frame = utils.resize_keep_ap(frame, height=800)
            cv2.imshow('frame', display_frame)

            # Keyboard controls
            k = cv2.waitKey(wait_time)

            if k == ord('q'): # quit
                break
            elif k == ord('s'): # Snapshot
                cv2.imwrite(f'save_{frame_cont}.png', display_frame)
            elif k == ord('g'):  # toggle gif
                if gif_buffer:
                    gifs_to_save[os.path.join(out_dir, 'gifs', f'gif_{frame_cont}.gif')] = gif_buffer
                    gif_buffer = None
                else:
                    print('Init gif')
                    gif_buffer = [gif_preprocess(frame)]
            elif k == ord('p'): # Pause
                wait_time_idx += 1
                # wait_time = int(not(bool(wait_time)))

        # Gif
        if gif_buffer:
            gif_buffer.append(gif_preprocess(frame))

        ret, frame = cap.read()
        frame_cont += 1

    cap.release()
    cv2.destroyAllWindows()

    # Save gifs
    if gifs_to_save:
        print('Saving gifs...')
        for path, buffer in gifs_to_save.items():
            print(path, '...')
            imageio.mimsave(path, buffer)

    # Compute APs
    with open(os.path.join(out_dir, 'AP_results.txt'), 'a') as fp:
        fp.write('\n=================\n')
        fp.write(f'AP {AP_thresh}\n')
        gt = detections[0]['rects']
        for det in detections[1:]:
            AP = utils.get_AP(gt, det['rects'], ovthresh=AP_thresh)
            fp.write(f'{det["full-name"]}: {AP}\n')

if __name__ == '__main__':
    main()