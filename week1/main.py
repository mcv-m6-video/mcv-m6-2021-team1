""" 
    MOTChallenge format [frame, ID, left, top, width, height, 1, -1, -1, -1].
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z> 
"""
import os
import cv2
import utils
import argparse
import pickle as pkl
import numpy as np

from matplotlib.pyplot import Figure
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

#PATHS
ANNOTATIONS_FILE = '../../data/ai_challenge_s03_c010-full_annotation.xml'
VIDEO_PATH = '../../data/AICity_data/train/S03/c010/vdo.avi'

DETECTIONS = {
    'rcnn': '../../data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt',
    'ssd': '../../data/AICity_data/train/S03/c010/det/det_ssd512.txt',
    'yolo': '../../data/AICity_data/train/S03/c010/det/det_yolo3.txt'
}


def main(mode, run_name, noisy_config=None, display=False, save=False):

    os.makedirs(f'runs/{run_name}/', exist_ok=True)

    gt_all_rects = utils.parse_xml_rects(ANNOTATIONS_FILE)


    if noisy_config:
        noisy_drop, noisy_pos, noisy_size, noisy_ar = noisy_config
        det_all_rects = utils.generate_noisy_bboxes(gt_all_rects, float(noisy_drop), float(noisy_pos),
            float(noisy_size), float(noisy_ar))
    else:
        det_all_rects = utils.parse_aicity_rects(DETECTIONS[mode])

    miou_over_time = []

    # Render video
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_cont = 0
    ret, frame = cap.read()
    wait_time = 1

    while(ret):
        gt_rects = gt_all_rects.get(f'f_{frame_cont}', None)
        det_rects = det_all_rects.get(f'f_{frame_cont}', None)

        if gt_rects:
            for r in gt_rects:
                frame = cv2.rectangle(frame, (int(r[0]), int(r[1])),  (int(r[2]), int(r[3])), (0, 255, 0), 2)

        if det_rects:
            for obj in det_rects:
                r = obj['bbox']
                frame = cv2.rectangle(frame, (int(r[0]), int(r[1])),  (int(r[2]), int(r[3])), (0, 0, 255), 2)

        if gt_rects and det_rects:
            miou = utils.get_frame_iou(gt_rects, det_rects)
            miou_over_time.append(miou)

        display_frame = cv2.resize(frame, tuple(np.int0(0.5*np.array(frame.shape[:2][::-1]))))

        if display:
            cv2.imshow('frame',display_frame)
            k = cv2.waitKey(wait_time)
            if k == ord('q'):
                break
            elif k == ord('s'):
                cv2.imwrite(f'save_{frame_cont}.png', display_frame)
            elif k == ord('p'):
                wait_time = int(not(bool(wait_time)))
        
        if save:
            # Was saving this for making the fancy gif, though it is too late now
            plt.figure()
            plt.plot(miou_over_time)
            plt.xlim([0, 2140])
            plt.ylim([0, 1])
            plt.savefig(f'runs/{run_name}/iou_plt_{frame_cont}.png')

            cv2.imwrite(f'runs/{run_name}/frame_{frame_cont}.png', frame)

        ret, frame = cap.read()
        frame_cont += 1

    cap.release()
    cv2.destroyAllWindows()

    # Compute mean metrics
    mAP = utils.get_AP(gt_all_rects, det_all_rects)
    mIOU = np.mean(miou_over_time)

    print(f'Mean statistics for {run_name}:\nmAP: {mAP}\nmIOU: {mIOU}')

    plt.figure()
    plt.plot(miou_over_time)
    plt.xlim([0, 2140])
    plt.ylim([0, 1])
    plt.xlabel('# Frame')
    plt.ylabel('mean IoU')
    plt.title(f'Mean IoU over time for {run_name} data')
    plt.savefig(f'runs/{run_name}_iou_plt_final.png')

    with open(f'runs/{run_name}_iou_raw.pkl', 'wb') as f:
        pkl.dump(miou_over_time, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', required=True, type=str)
    parser.add_argument('-n', '--name', required=True, type=str)
    parser.add_argument('-d', '--display', action='store_const', const=True, default=False)
    parser.add_argument('-s', '--save', action='store_const', const=True, default=False)
    parser.add_argument('--noise', type=str, help='Format drop-pos-size-ar')

    args = parser.parse_args()

    noisy_config = args.noise.split('-') if args.noise else None
    main(args.mode, args.name, display=args.display, save=args.save, noisy_config=noisy_config)