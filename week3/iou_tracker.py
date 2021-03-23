import utils
from munkres import Munkres


# 2141 frames in total
TOTAL_FRAMES = 2141
TOTAL_FRAMES = 200

VIDEO_PATH = "../data/AICity_data/train/S03/c010/vdo.avi"
# GT_RECTS_PATH = "../data/ai_challenge_s03_c010-full_annotation.xml"
# AI_GT_RECTS_PATH = "../data/AICity_data/train/S03/c010/gt/gt.txt"
DETECTIONS = 'm6-aicity_retinanet_R_50_FPN_3x_rp128.txt'

tracked_object_dic = {}
TRACKS_COUNTER = 0

class tracked_object:

    def __init__(self, idd, bbox, tracker_life = 5, min_iou_th = 0.5):
        self.id = idd
        self.bbox = bbox
        self.tracker_life = tracker_life
        self.min_iou_th = min_iou_th
        _add_track(self)

    def _add_track(self):
        tracked_object_list.append(self)
        return


def adj_track(det_bbox):
        # to access to iou_matrix: iou_matrix[detector_index][tracker_index]
            best_iou = 0
            idx = -1

            for id_t in tracked_object_dic:
                to = tracked_object_dic[id_t]
                iou = get_rect_iou(to.bbox, det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    idx = track_obj.id
                
            if idx != -1 and iou > 0.5:
                 track_obj.tracker_life  = 5

            else:
                track_obj(bbox)
                tracked_object_dic[tracked_object_dic]
                TRACKS_COUNTER+=1
               

                
        # iou_matrix = [[1 - get_rect_iou(t_det.bbox, det_bbox) for t_det in tracker_bboxes]
        #                 for det_bbox in detector_bboxes]

        idxs = Munkres().compute(iou_matrix)
    return id

def main():

    det_rects = utils.parse_aicity_rects('m6-aicity_retinanet_R_50_FPN_3x_rp128.txt')
    
    #{'bbox': [1190.155517578125, 102.54630279541016, 1341.8289794921875, 179.31149291992188], 'conf': 0.070556417107582, 'id': -1},
    for f in range(TOTAL_FRAMES):
        det_rects[f'f_{f}'] #detections per frame
        for det in det_rects[f'f_{f}']:
            det['id'] = adj_track(det['bbox'])

        print(det_rects[f'f_{f}'])
    #gt_rects_detformat = {f: [{'bbox': r, 'conf':1} for r in v] for f, v in gt_rects.items()}

main()
