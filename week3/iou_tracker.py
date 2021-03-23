import utils
# from munkres import Munkres


# 2141 frames in total
TOTAL_FRAMES = 2141
TOTAL_FRAMES = 10

VIDEO_PATH = "../data/AICity_data/train/S03/c010/vdo.avi"
GT_RECTS_PATH = "../../Data/ai_challenge_s03_c010-full_annotation.xml"
# AI_GT_RECTS_PATH = "../data/AICity_data/train/S03/c010/gt/gt.txt"
DETECTIONS = 'm6-aicity_retinanet_R_50_FPN_3x_rp128.txt'

tracked_object_dic = {}


class tracked_object:
    TRACKS_COUNTER = 0
    def __init__(self, idd, bbox, tracker_life = 5):
        self.id = idd
        self.bbox = bbox
        self.tracker_life = tracker_life
        self.not_live = 0
        self._add_track()

    def _add_track(self):
        tracked_object_dic[self.id] = self
        return


def adj_track(det_bbox):
      
    best_iou = 0
    idx = -1

    for id_t in tracked_object_dic:

        iou = utils.get_rect_iou(tracked_object_dic[id_t].bbox, det_bbox)    
        if iou > best_iou:
            best_iou = iou
            idx = id_t

    # print('BEST: ',idx)
    # print('iou: ', best_iou)
    if int(idx) != -1 and best_iou > 0.5:
        tracked_object_dic[idx].tracker_life = 5
        tracked_object_dic[idx].bbox = det_bbox
    else:
        idx = str(tracked_object.TRACKS_COUNTER)
        tracked_object(idx, det_bbox)
        tracked_object.TRACKS_COUNTER+=1
    return idx

def decrease_memory():
    deleting_list = []
    for idx in tracked_object_dic:
        if tracked_object_dic[idx].tracker_life > 0:
            tracked_object_dic[idx].tracker_life -= 1
        else:
            deleting_list.append(idx)
        
    for idx in deleting_list:
        del tracked_object_dic[idx]

def get_idf1(det_rects, gt_rects):

    tracks_dic = {}

    score_dic = {
        'id_det' : -1,
        'total' : 0,
        'tp': 0,
        'fp': 0,
        'fn': 0
    }



    # for f in gt_rects:
    #     for tracks in gt_rects[f]:
    #         if track['id'] not in tracks_dic:
    #             tracks_dic[tracks['id']] = score_dic

    #         best_iou = 0
    #         id_t = -1
    #         for t_det in det_rects[f]:

    #             iou = utils.get_rect_iou(tracks['bbox'], t_det['bbox'])
    #             if iou > best_iou:
    #                 best_iou = iou
    #                 id_t = t_det['id']
    #         if id_t == tracks_dic[tracks['id']]['id_det']
    #         #tracks_dic[tracks['id']]['id_det'] = id_t


    



def main():

    det_rects = utils.parse_aicity_rects('m6-aicity_retinanet_R_50_FPN_3x_rp128.txt')
    
    #{'bbox': [1190.155517578125, 102.54630279541016, 1341.8289794921875, 179.31149291992188], 'conf': 0.070556417107582, 'id': -1},
    order = sorted(det_rects, key=lambda x: int(x[2:]) )
    for f in order:
        for det in det_rects[f]:
            if det['conf'] > 0.5:
                det['id'] = adj_track(det['bbox'])
            # print(det)
        decrease_memory()
        # print('NEW FRAME:', len(tracked_object_dic))
    utils.save_aicity_rects('test.txt', det_rects)
    print(len(tracked_object_dic))

    gt_rects = utils.parse_aicity_rects(GT_RECTS_PATH)
    get_idf1(det_rects, gt_rects)

if __name__ == '__main__':
    dt = utils.parse_xml_rects(GT_RECTS_PATH)
    utils.save_aicity_rects('full_gt.txt', dt)
   # main()