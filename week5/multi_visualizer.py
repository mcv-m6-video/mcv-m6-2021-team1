import os
import sys
import datetime
import cv2
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pygifsicle import optimize


currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import utils as w5utils
from week3 import utils as w3utils

DATA_PATH = '/home/capiguri/code/datasets/m6data'
FRAME_NUM_PATH = os.path.join(DATA_PATH, 'cam_framenum')
TIMESTAMP_PATH = os.path.join(DATA_PATH, 'cam_timestamp')

GIF_OUT_DIR = './w6gifs/'

dets2use = ['mask_rcnn']
tracks2use = ['mtsc_deepsort_mask_rcnn']

class MOTCamera():
    """
    Object representing a camera in the MOTSChallenge format
    """
    def __init__(self, path, det='mask_rcnn', track='deepsort_mask_rcnn'):
        self.cam_name = os.path.split(path)[-1]
        print(f'\tCreating object for camera {self.cam_name}...')

        self.gt = w3utils.parse_aicity_rects(os.path.join(path, 'gt', 'gt.txt'))
        self.detections = { k[4:-4]: w3utils.parse_aicity_rects(os.path.join(path, 'det', k))
            for k in os.listdir(os.path.join(path, 'det')) if '.txt' in k and any(t in k for t in dets2use)}
        self.trackings = { k[5:-4]: w3utils.parse_aicity_rects(os.path.join(path, 'mtsc', k))
            for k in os.listdir(os.path.join(path, 'mtsc')) if '.txt' in k and any(t in k for t in tracks2use)}
        self.num_frames = 0
        self.videopath = os.path.join(path, 'vdo.avi')
        self.roi = cv2.imread(os.path.join(path, 'roi.jpg'))
        self.FPS = 10 if 'c015' not in path else 8
        
        self.det = det
        self.track = track

        self.last_frame = None

    def init_capture(self):
        print(f'\tInitializing capture for camera {self.cam_name}...')
        self.frame_cont = 0
        self.cap = cv2.VideoCapture(self.videopath)

    def get_frame(self):
        print(f'\tGetting frame {self.frame_cont} frame for camera {self.cam_name}...')
        self.frame_cont += 1
        ret, frame =  self.cap.read()
        frame = self._paint_rects(frame)
        self.last_frame = frame
        return ret, frame

    def _paint_rects(self, im):
        # det = self.detections[self.det]
        # det = self.trackings[self.track]
        det = self.gt
        frame = w3utils.pretty_rects(im, det.get(f'f_{self.frame_cont}', []), 'test', (0,255,0),
                conf_thresh=0.7, tracking = True)
        return im
    
    def get_last_frame(self):
        return self.last_frame

    def _close_vid(self):
        self.cap.release()

    def __del__(self):
        self._close_vid()


class MOTSequence():
    """
    Object representing a sequence in the MOTSChallenge format
    """
    def __init__(self, seq_num, cam_ids=[], det='mask_rcnn', track='deepsort_mask_rcnn'):
        if not cam_ids:
            cam_ids = list(range(1000)) # dirty ugly patch but whatever

        print(f'Creating object for sequence S{str(seq_num).zfill(2)}...')
        path = os.path.join(DATA_PATH, 'train', f'S{str(seq_num).zfill(2)}')
        self.cams = {p: MOTCamera(os.path.join(path, p), det=det, track=track) for p in os.listdir(path) if int(p[1:]) in cam_ids}
        self.num_cams = len(self.cams)
        self.timestamps = {}
        with open(os.path.join(TIMESTAMP_PATH, f'S{str(seq_num).zfill(2)}.txt'), 'r') as f:
            for l in f:
                k, v = l.split()
                self.timestamps[k] = float(v)
        
        self.gif_buffer = {k: [] for k in self.cams}
        self.gifs_to_save = {k: {} for k in self.cams}
        self.gif_flag = False
        self.dtstring = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

        # Add our detections
        for cam in os.listdir(os.path.join('mtrackings',f'S{str(seq_num).zfill(2)}')):
            for algorithm in os.path.join('mtrackings',f'S{str(seq_num).zfill(2)}', cam):
                self.cams[cam].trackings.append(
                    w3utils.parse_aicity_rects(os.path.join('mtrackings',f'S{str(seq_num).zfill(2)}', cam, algorithm), zero_index=0)
                )

    def init_visualize(self, ids=None):
        self.frame_cont = 0

        for c, cam in self.cams.items():
            cam_id = int(c[1:])
            if ids and cam_id not in ids:
                continue
            cam.init_capture()
    
    def next_frame(self, ids=None, txt_info='', scale=0.5):

        frames = {}

        for c, cam in self.cams.items():
            cam_id = int(c[1:])
            if ids and cam_id not in ids:
                continue

            if self.frame_cont >= self.timestamps[c]*cam.FPS:
                ret, frame = cam.get_frame()
                frames[c] = frame if ret else np.zeros((100,100), dtype=np.uint8)

        self.frame_cont += 1

        if self.gif_flag:
            for k, v in self.frames:
                gif_buffer[k].append(v)

        self._show_frames(frames, txt_info=txt_info, scale=scale)

    def _show_frames(self, frames, txt_info='', scale=0.5):
        for k, v in frames.items():
            v = cv2.resize(v, tuple(np.int0(scale*np.array(v.shape[:2][::-1]))))

            h, w = v.shape[:2]
            x, y = int(w*0.75), int(h*0.8)
            txt_info += '[REC]' if self.gif_flag else '[___]'
            v = cv2.putText(v, txt_info, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                w3utils.get_optimal_font_scale(txt_info, w*0.1), (0, 0, 255), 2)

            cv2.imshow(k, v)
        # return cv2.waitKey(0)
    
    def __del__(self):
        for _,c in self.cams.items():
            del c
    
    def toggle_gifs(self):
        if self.gif_flag:
            for k, v in self.cams:
                self.gif_flag = False
                self.gifs_to_save[k][os.path.join(GIF_OUT_DIR, self.dtstring, f'gif_{v.cam_name}_{v.frame_cont}.gif')] = gif_buffer
                self.gif_buffer[k] = None
        else:
            print('Init gif')
            self.gif_flag = True
            for k, v in self.cams.items():
                self.gif_buffer[k] = [w5utils.gif_preprocess(v.get_last_frame())]


os.makedirs(GIF_OUT_DIR, exist_ok=True)

s = MOTSequence(3, cam_ids=[13, 14, 15])
s.init_visualize()

WAIT_TIME_LIST = [30, 60, 0, 10] # Available display speeds
wait_time_idx = 0 # start at 33 FPS (15 ms per frame)
disp_scale = 0.5

k = True
while k != ord('q'):
    wait_time = WAIT_TIME_LIST[wait_time_idx % len(WAIT_TIME_LIST)]
    FPS = int(1e3/wait_time if wait_time != 0 else 0)

    s.next_frame(txt_info=f'{FPS} FPS', scale=disp_scale)

    # Keyboard controls
    k = cv2.waitKey(wait_time)

    if k == ord('q'): # quit
        break
    elif k == ord('+'):
        disp_scale += 0.1 if disp_scale < 1 else 0
    elif k == ord('-'):
        disp_scale -= 0.1 if disp_scale > 0.1 else 0
    # elif k == ord('s'): # Snapshot
    #     cv2.imwrite(f'save_{frame_cont}.png', display_frame)
    elif k == ord('g'):  # toggle gif
        s.toggle_gifs()
    elif k == ord('p'): # Pause
        wait_time_idx += 1
        # wait_time = int(not(bool(wait_time)))
    
print(s.gifs_to_save)