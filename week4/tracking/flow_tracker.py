
import numpy as np
import cv2
import statistics


class GFFlowTracker:

    def __init__(self, strategy="mean"):
        self.bbox = None
        self.prev_frame = None
        self.flow = None
        self.strategy = strategy


    def init(self, frame, bbox):
        self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self.bbox = bbox

    def update(self, frame):
        """
        Returns the predicted bbox from the GF optical flow estimation
        """
        self.__update_state(frame)

        if self.strategy == "mean":
            ax = np.mean(np.ravel(self.flow[...,0]))
            ay = np.mean(np.ravel(self.flow[...,1]))
        else:
            ax = statistics.median(np.ravel(self.flow[...,0]))
            ay = statistics.median(np.ravel(self.flow[...,1]))

        P = 1
        new_x = int(self.bbox[0] + self.bbox[2] * ax * P)
        new_y = int(self.bbox[1] + self.bbox[3] * ay * P)
        self.bbox = (new_x, new_y, self.bbox[2], self.bbox[3])
        return True, self.bbox

    def __update_state(self, frame):
        x0, y0, w, h = self.bbox
        x, y = x0 + w, y0 + h
        h_m = int(0.1*h) # some context information helps the OF computation
        w_m = int(0.1*w)
        x0, y0 = max(0, x0 - w_m), max(0, y0 - h_m)
        x, y = min(frame.shape[1], x + w_m), min(frame.shape[0], y + h_m)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(self.prev_frame[y0:y, x0:x],frame[y0:y, x0:x], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev_frame = frame
        self.flow = flow




feature_params = dict( maxCorners = 100,
                    qualityLevel = 0.3,
                    minDistance = 7,
                    blockSize = 7 )
lk_params = dict( winSize = (15,15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

class LKFlowTracker:

    def __init__(self, strategy="median"):
        self.bbox = None
        self.prev_frame = None
        self.p0 = None
        self.p1 = None
        self.strategy = strategy


    def init(self, frame, bbox):
        self.bbox = bbox
        bbox = (bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1])

        self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        x0, y0, x, y = bbox

        self.p0 = cv2.goodFeaturesToTrack(self.prev_frame[y0:y, x0:x], **feature_params)
        self.p1 = self.p0

    def update(self, frame):
        """
        Returns the predicted bbox from the LK optical flow estimation
        """
        self.__update_state(frame)

        if self.strategy == "median":
            ax = statistics.median(self.p1[:,:,0] - self.p0[:,:,0])
            ay = statistics.median(self.p1[:,:,1] - self.p0[:,:,1])
        else:
            ax = np.mean(self.p1[:,:,0] - self.p0[:,:,0])
            ay = np.mean(self.p1[:,:,1] - self.p0[:,:,1])

        P = 1
        new_x = int(self.bbox[0] + self.bbox[2] * ax * P)
        new_y = int(self.bbox[1] + self.bbox[3] * ay * P)
        return True, (new_x, new_y, self.bbox[2], self.bbox[3])

    def __update_state(self, frame):
        x0, y0, w, h = self.bbox
        x, y = x0 + w, y0 + h

        self.p0 = self.p1
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame[y0:y, x0:x], frame[y0:y, x0:x], self.p0, None, **lk_params)
        self.prev_frame = frame
        self.p1 = p1[st==1].reshape(-1,1,2)
        self.p0 = self.p0[st==1].reshape(-1,1,2)
