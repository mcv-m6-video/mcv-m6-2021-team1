

import sys
sys.path.append("./week3/kalman/pysot")
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.model_load import load_pretrain
from week3.kalman.static_tracker import StaticTracker
from week3.kalman.kalman_tracker import KalmanTracker
import torch
import cv2


OPENCV_TRACKERS = {
    # OpenCV trackers
    "kcf":          cv2.TrackerKCF_create,
    "csrt":         cv2.TrackerCSRT_create,
}

PYSOT_TRACKERS = {
    "siamrpn_r50":      "siamrpn_r50_l234_dwxcorr",
    "siamrpn_alex":     "siamrpn_alex_dwxcorr",
    "siamrpn_mobile":   "siamrpn_mobilev2_l234_dwxcorr",
    "siammask":         "siammask_r50_l3",
}


def load_pysot_model(tracker_type):
    configpath = "./week3/kalman/pysot/experiments/" + PYSOT_TRACKERS[tracker_type] + \
                 "/config.yaml"
    modelpath = "./week3/kalman/pysot/models/" + PYSOT_TRACKERS[tracker_type] + ".pth"

    cfg.merge_from_file(configpath)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # load model
    model = ModelBuilder()
    model.load_state_dict(torch.load(modelpath, map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)
    return load_pretrain(model, modelpath).cuda().eval()


# Class that encapsulates all SINGLE trackers that we use
class SingleTracker:

    def __init__(self, tr):
        self.type = tr
        self.PYSOT_TRACKER_THRESHOLD = 0.3

        if tr in OPENCV_TRACKERS:
            self.tracker = OPENCV_TRACKERS[tr]()
        elif tr in PYSOT_TRACKERS:
            self.tracker = build_tracker(load_pysot_model(self.type))
        elif tr == "kalman":
            self.tracker = KalmanTracker()
        elif tr == "iou":
            self.tracker = StaticTracker()
        else:
            raise Exception("Tracker not supported")

    def init(self, frame, bbox):
        if self.tracker is None:
            return

        # we adjust the bbox in case it gets out of the frame limits
        adjusted_bbox = (max(bbox[0], 0),
                         max(bbox[1], 0),
                         min(bbox[2], frame.shape[1] - bbox[0]),
                         min(bbox[3], frame.shape[0] - bbox[1]))
        return self.tracker.init(frame, adjusted_bbox)

    # Returns (success, bbox) with bbox as (x0, y0, x, y)
    def track(self, frame):
        if self.type in PYSOT_TRACKERS:
            tracked = self.tracker.track(frame)
            (success, (x, y, w, h)) = tracked['best_score'] > self.PYSOT_TRACKER_THRESHOLD, list(map(int, tracked['bbox']))
        else:
            (success, (x, y, w, h)) = self.tracker.update(frame)
        return success, (x, y, x + w, y + h)

    def update_state(self, bbox):
        if self.type == "kalman":
            self.tracker.update_state(bbox)