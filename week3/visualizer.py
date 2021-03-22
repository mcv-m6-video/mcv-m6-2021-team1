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
import cv2
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt

import utils
import detection

def gif_preprocess(im, width=512):
    im = utils.resize_keep_ap(im, width=width)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

TOTAL_FRAMES = 2141
VIDEO_PATH = "../../data/AICity_data/train/S03/c010/vdo.avi"
GT_RECTS_PATH = "../../data/ai_challenge_s03_c010-full_annotation.xml"
AI_GT_RECTS_PATH = "../../data/AICity_data/train/S03/c010/gt/gt.txt"
OUT_DIR = 'visualizer'

# Load detections
detections = [
    {
        'name': 'gt',
        'color': (0, 255, 0),
        'rects': utils.parse_xml_rects(GT_RECTS_PATH)
    },
    {
        'name': 'aigt',
        'color': (0, 0, 255),
        'rects': utils.parse_aicity_rects(AI_GT_RECTS_PATH)
    },
]

def main(display=True):

    # Create output dirs
    os.makedirs(os.path.join(OUT_DIR, 'snapshots'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'gifs'), exist_ok=True)

    # Render video
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
            frame = utils.pretty_rects(frame, det['rects'].get(f'f_{frame_cont}', []), det['name'], det['color'])
 
        if display:
            # Resize
            display_frame = utils.resize_keep_ap(frame, height=800)

            # Display speed
            wait_time = WAIT_TIME_LIST[wait_time_idx % len(WAIT_TIME_LIST)]
            os.system('clear')
            print(f'Speed: {1e3/wait_time if wait_time != 0 else 0} FPS')

            # Display
            cv2.imshow('frame', display_frame)

            # Keyboard controls
            k = cv2.waitKey(wait_time)

            if k == ord('q'): # quit
                break
            elif k == ord('s'): # Snapshot
                cv2.imwrite(f'save_{frame_cont}.png', display_frame)
            elif k == ord('g'):  # toggle gif
                if gif_buffer:
                    gifs_to_save[os.path.join(OUT_DIR, 'gifs', f'gif_{frame_cont}.gif')] = gif_buffer
                    gif_buffer = None
                else:
                    print('Init gif')
                    gif_buffer = [gif_preprocess(frame)]
            elif k == ord('p'): # Pause
                wait_time_idx += 1
                # wait_time = int(not(bool(wait_time)))

        # Gif
        if gif_buffer:
            print('Update gif')
            gif_buffer.append(gif_preprocess(frame))

        ret, frame = cap.read()
        frame_cont += 1

    cap.release()
    cv2.destroyAllWindows()

    # Save gifs
    print('Saving gifs...')
    for path, buffer in gifs_to_save.items():
        print(path, '...')
        imageio.mimsave(path, buffer)


if __name__ == '__main__':
    main()