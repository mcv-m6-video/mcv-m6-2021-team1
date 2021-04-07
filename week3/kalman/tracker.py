import collections
from week3.kalman.single_tracker import SingleTracker
from munkres import Munkres
from week3.kalman.tracking_utils import compute_intersection_over_union


Detection = collections.namedtuple('Detection', 'track_id bbox score from_tracker')


class TracksManager:

    def __init__(self, tracker_type = "identity", tracker_life = 10, min_iou_th = 0.5):
        self.lost_tracker_life = tracker_life
        self.tracker_type = tracker_type
        self.min_iou_th = min_iou_th

        # info about the active trackers
        self.trackers = {}  # dict as (key, value) = (id, tracker)
        self.trackid2framesinactive = {}

        # how many trackers are active at the moment
        self.TRACKS_COUNTER = 0

    def update(self, frame, detector_bboxes):
        # on every trackers update, trackers inactive frames is added 1 (will be reset later if found)
        for t_id in list(self.trackers.keys()):
            self.__add_tracker_inactive_frame(t_id)

        # prepare the results structure with only successed tracking
        results = [(tr_id, self.trackers[tr_id].track(frame)) for tr_id in self.trackers]
        tracker_bboxes = [Detection(track_id=tr_id, bbox=bbox, score=1, from_tracker=True)
                for (tr_id, (success, bbox)) in results if success]

        # update of the trackers and transform to standard format
        return self.__associate_detections(frame, detector_bboxes, tracker_bboxes)

    # we add a new object to be tracked received in format (XYXY)
    def __add(self, frame, bbox, track_id=-1):
        # we set the box to be tracked
        tracker_box = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
        if track_id == -1:
            # we set a new track id
            track_id = self.TRACKS_COUNTER
            self.TRACKS_COUNTER += 1
        elif track_id in self.trackers.keys():
            return False  # already exists

        # we create the new tracker
        new_tracker = SingleTracker(self.tracker_type)
        new_tracker.init(frame, tracker_box)

        # we save and initialize it
        self.trackers[track_id] = new_tracker
        self.trackid2framesinactive[track_id] = 0

        return track_id

    # Called every time the tracker finds the object -> start tracking again with the new detection as initial object sample
    def __reset_tracker(self, frame, tr_id, det_bbox):
        self.__remove_tracker(tr_id)
        self.__add(frame, det_bbox, tr_id)

    def __found_tracker(self, tr_id):
        self.trackid2framesinactive[tr_id] = 0

    # Tracker is removed from the list of trackers (when killed, for example)
    def __remove_tracker(self, track_id):
        if track_id not in self.trackers:
            return
        self.trackers.pop(track_id, None)

        if track_id not in self.trackid2framesinactive:
            return
        self.trackid2framesinactive.pop(track_id, None)

    def __add_tracker_inactive_frame(self, track_id):
        self.trackid2framesinactive[track_id] += 1
        # if its live has ended, kill it
        if self.trackid2framesinactive[track_id] > self.lost_tracker_life:
            self.__remove_tracker(track_id)

    def __associate_detections(self, frame, detector_bboxes, tracker_bboxes):
        # for every tracker box, we find the corresponding detection
        # if no detection corresponds, we keep it because it can be an occlusion or FN => will wait X frames to find it
        results = []
        track_ids_associated = []

        # to access to iou_matrix: iou_matrix[detector_index][tracker_index]
        if len(detector_bboxes) > 0:
            iou_matrix = [[1 - compute_intersection_over_union(t_det.bbox, det_bbox) for t_det in tracker_bboxes]
                            for det_bbox in detector_bboxes]
            indexes = Munkres().compute(iou_matrix)

            # we must include all detections: {associated} U {non-associated}
            # STEP 1. first associated detector => set tracker already running
            associated_det_indexs = []
            for (det_index, tr_index) in indexes:
                det_bbox = detector_bboxes[det_index]
                score = det_bbox[4]
                det_bbox = [det_bbox[i] for i in range(4)]
                track_id = tracker_bboxes[tr_index].track_id
                # if over threshold, correspondence found
                if 1 - iou_matrix[det_index][tr_index] >= self.min_iou_th:
                    results.append(Detection(track_id=track_id, bbox=det_bbox, score=score,
                                                from_tracker=False))
                    # for debug purposes: we show the tracker detection too
                    results.append(Detection(track_id=track_id, bbox=tracker_bboxes[tr_index].bbox, score=score,
                                                from_tracker=True))
                    track_ids_associated.append(track_id)
                    associated_det_indexs.append(det_index)

                    # Tracked object was found! => reset with the detection box (which should be more accurate)
                    if self.tracker_type == "kalman":
                        self.trackers[track_id].update_state(frame, det_bbox)
                        self.__found_tracker(track_id) # we do not reset it for performance issues
                    elif "siam" not in self.tracker_type.lower() or 1 - iou_matrix[det_index][tr_index] < 0.85:
                        self.__reset_tracker(frame, track_id, det_bbox)
                    else:
                        self.__found_tracker(track_id) # we do not reset it for performance issues

            # STEP 2. now non-associated detector => new tracker (so new track id)
            non_associated = [i for i in range(len(detector_bboxes)) if i not in associated_det_indexs]
            for index in non_associated:
                det_bbox = detector_bboxes[index]
                score = det_bbox[4]
                det_bbox = [det_bbox[i] for i in range(4)]
                track_id = self.__add(frame, det_bbox)
                results.append(Detection(track_id=track_id, bbox=det_bbox, score=score,
                                            from_tracker=False))

        # STEP 3. we add the not associated tracker results (possible occlusions that the detector does not detect)
        not_associated_detections = filter(lambda det: det.track_id not in track_ids_associated, tracker_bboxes)
        for det in not_associated_detections:
            results.append(Detection(track_id=det.track_id, bbox=det.bbox, score=0,
                                        from_tracker=True))
        return results
