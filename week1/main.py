""" 
    MOTChallenge format [frame, ID, left, top, width, height, 1, -1, -1, -1].
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z> 
"""
import os
import cv2
import numpy as np
import utils
import pickle as pkl
from matplotlib.pyplot import Figure
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

#PATHS
annotations_file = '../../data/ai_challenge_s03_c010-full_annotation.xml'
VIDEO_PATH = '../../data/AICity_data/train/S03/c010/vdo.avi'
GT_PATH = '../../data/AICity_data/train/S03/c010/gt/gt.txt'

RUN_NAME = 'rcnn-plot-test'
DET_PATH = '../../data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt'

RUN_NAME = 'ssd-low-pa'
DET_PATH = '../../data/AICity_data/train/S03/c010/det/det_ssd512.txt'

# RUN_NAME = 'yolo-low-pa'
# DET_PATH = '../../data/AICity_data/train/S03/c010/det/det_yolo3.txt'

RUN_NAME = 'noisy-none'
noisy = False
show = True
save=False

def main():

    if not os.path.exists(f'runs/{RUN_NAME}/'):
        os.mkdir(f'runs/{RUN_NAME}/')

    gt_all_rects = utils.parse_xml_rects(annotations_file)

    tol_dropout = 0.
    std_pos = 0.
    std_size = 0.
    std_ar = 0.

    if noisy:
        det_all_rects = utils.generate_noisy_bboxes(gt_all_rects, tol_dropout, std_pos, std_size, std_ar)
    else:
        det_all_rects = utils.parse_aicity_rects(DET_PATH)

    # Render video
    cap = cv2.VideoCapture(VIDEO_PATH)
    miou_over_time = []
    map_over_time = []
    frame_cont = 0
    ret, frame = cap.read()

    wait_time = 1
    while(ret):
    
        gt_rects = gt_all_rects.get(frame_cont, None)
        det_rects = det_all_rects.get(frame_cont, None)

        if gt_rects:
            for r in gt_rects:
                frame = cv2.rectangle(frame, (int(r[0]), int(r[1])),  (int(r[2]), int(r[3])), (0, 255, 0), 2)

        if det_rects:
            for r in det_rects:
                frame = cv2.rectangle(frame, (int(r[0]), int(r[1])),  (int(r[2]), int(r[3])), (0, 0, 255), 2)

        # print('Frame',  frame_cont)
        if gt_rects and det_rects: # If we can compute the metrics, just skip
            miou = utils.get_frame_iou(gt_rects, det_rects)
            miou_over_time.append(miou)
            ap = utils.get_AP(gt_rects, det_rects)
            map_over_time.append(ap)
            # print('iou:', miou, 'ap:', ap)
        
        display_frame = cv2.resize(frame, tuple(np.int0(0.5*np.array(frame.shape[:2][::-1]))))

        if show:
            cv2.imshow('frame',display_frame)
            k = cv2.waitKey(wait_time)
            if k == ord('q'):
                break
            elif k == ord('s'):
                cv2.imwrite(f'save_{frame_cont}.png', display_frame)
            elif k == ord('p'):
                wait_time = int(not(bool(wait_time)))
        
        if save:
            plt.figure()
            plt.plot(miou_over_time)
            plt.xlim([0, 2140])
            plt.ylim([0, 1])
            plt.savefig(f'runs/{RUN_NAME}/iou_plt_{frame_cont}.png')

            plt.figure()
            plt.plot(map_over_time)
            plt.xlim([0, 2140])
            plt.ylim([0, 1])
            plt.savefig(f'runs/{RUN_NAME}/map_plt_{frame_cont}.png')

            cv2.imwrite(f'runs/{RUN_NAME}/frame_{frame_cont}.png')


        ret, frame = cap.read()
        frame_cont += 1

    cap.release()
    cv2.destroyAllWindows()

    mAP = np.mean(map_over_time)
    mIOU = np.mean(miou_over_time)

    print(f'Mean statistics for {RUN_NAME}:\nmAP: {mAP}\nmIOU: {mIOU}')

    plt.figure()
    plt.plot(miou_over_time)
    plt.xlim([0, 2140])
    plt.ylim([0, 1])
    plt.xlabel('# Frame')
    plt.ylabel('mean IoU')
    plt.title(f'Mean IoU over time for {RUN_NAME} data')
    plt.savefig(f'runs/{RUN_NAME}_iou_plt_final.png')

    with open(f'runs/{RUN_NAME}_iou_raw.pkl', 'wb') as f:
        pkl.dump(miou_over_time, f)

    plt.figure()
    plt.plot(map_over_time)
    plt.xlim([0, 2140])
    plt.ylim([0, 1])
    plt.xlabel('# Frame')
    plt.ylabel('mean AP')
    plt.title('Mean Average Precision over time for {RUN_NAME} data')
    plt.savefig(f'runs/{RUN_NAME}_map_plt_final.png')

    with open(f'runs/{RUN_NAME}_map_raw.pkl', 'wb') as f:
        pkl.dump(map_over_time, f)

main()