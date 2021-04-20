import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from week3 import utils as w3utils

DATA_PATH = '/home/jchaves/code/temp/m6data/'
FRAME_NUM_PATH = os.path.join(DATA_PATH, 'cam_framenum')
TIMESTAMP_PATH = os.path.join(DATA_PATH, 'cam_timestamp')

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

    def init_capture(self):
        print(f'\tInitializing capture for camera {self.cam_name}...')
        self.frame_cont = 0
        self.cap = cv2.VideoCapture(self.videopath)

    def get_frame(self):
        print(f'\tGetting frame {self.frame_cont} frame for camera {self.cam_name}...')
        self.frame_cont += 1
        ret, frame =  self.cap.read()
        frame = self._paint_rects(frame)
        return ret, frame

    def _paint_rects(self, im):
        # det = self.detections[self.det]
        # det = self.trackings[self.track]
        det = self.gt
        frame = w3utils.pretty_rects(im, det.get(f'f_{self.frame_cont}', []), 'test', (0,255,0),
                conf_thresh=0.7, tracking = det.get('tracking', False))
        return im

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
        self.cams= {p: MOTCamera(os.path.join(path, p), det=det, track=track) for p in os.listdir(path) if int(p[1:]) in cam_ids}
        self.num_cams = len(self.cams)
        self.timestamps = {}
        with open(os.path.join(TIMESTAMP_PATH, f'S{str(seq_num).zfill(2)}.txt'), 'r') as f:
            for l in f:
                k, v = l.split()
                self.timestamps[k] = float(v)

    def init_visualize(self, ids=None):
        self.frame_cont = 0

        for c, cam in self.cams.items():
            cam_id = int(c[1:])
            if ids and cam_id not in ids:
                continue
            cam.init_capture()
    
    def next_frame(self, ids=None):

        frames = {}

        for c, cam in self.cams.items():
            cam_id = int(c[1:])
            if ids and cam_id not in ids:
                continue

            if self.frame_cont >= self.timestamps[c]*cam.FPS:
                ret, frame = cam.get_frame()
                frames[c] = frame if ret else np.zeros((100,100), dtype=np.uint8)

        self.frame_cont += 1
        return self._show_frames(frames)

    def _show_frames(self, frames):
        for k, v in frames.items():
            v = cv2.resize(v, tuple(np.int0(0.5*np.array(v.shape[:2][::-1]))))
            cv2.imshow(k, v)
        return cv2.waitKey(0)
    
    def __del__(self):
        for _,c in self.cams.items():
            del c


s = MOTSequence(3, cam_ids=[10, 11, 12, 13, 14, 15])
s.init_visualize()

k = True
while k != ord('q'):
    k = s.next_frame()

