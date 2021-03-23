
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import imageio
import argparse
import pandas as pd
from models import GaussianModel, AdaptiveGaussianModel

import utils
import detection
import random
from tqdm import tqdm

# 2141 frames in total
TOTAL_FRAMES = 2141
#PERCENTAGE = 0.25
#ALPHA = 11
#P = 0.001

# VIDEO_PATH = "../../AICity_data/train/S03/c010/vdo.avi"
VIDEO_PATH = "../../data/AICity_data/train/S03/c010/vdo.avi"
GT_RECTS_PATH = "../../data/ai_challenge_s03_c010-full_annotation.xml"

def main(args, evaluate=False):

    if not os.path.exists(VIDEO_PATH):
        print("Video does not exist.")
        return

    model_frames = int(args.percentage * TOTAL_FRAMES)

    if args.model == "gm":
        model = GaussianModel(VIDEO_PATH, model_frames, args.alpha, \
                                checkpoint=f"{args.colorspace}_{args.percentage}", colorspace=args.colorspace)
        MODEL_NAME = "GaussianModel"
        results_path = f"results/{MODEL_NAME}/{args.colorspace}_{args.alpha}_{args.percentage}"
    elif args.model == "agm":
        model = AdaptiveGaussianModel(VIDEO_PATH, model_frames, args.alpha, args.p, \
                                checkpoint=f"{args.colorspace}_{args.percentage}", colorspace=args.colorspace)
        MODEL_NAME = "AdaptiveGaussianModel"
        results_path = f"results/{MODEL_NAME}/{args.colorspace}_{args.alpha}_{args.p}_{args.percentage}"
    else:
        raise Exception

    model.model_background()

    if not evaluate:
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        writer = imageio.get_writer(f"{results_path}/video.mp4", fps=25)

    foreground, I = model.compute_next_foreground()
    if not evaluate:
        writer.append_data(foreground)
    counter = int(TOTAL_FRAMES*args.percentage)
    det_rects = {}
    gt_rects = utils.parse_xml_rects(GT_RECTS_PATH, True) 
    gt_rects_detformat = {f: [{'bbox': r, 'conf':1} for r in v] for f, v in gt_rects.items()}

    while foreground is not None:
        foreground, recs = detection.post_processing(foreground)

        det_rects[f'f_{counter}'] = recs

        #if args.display:
        #    utils.imshow_rects(I, [{'rects': recs, 'color': (0,0,255)}, 
        #        {'rects': gt_rects_detformat.get(f'f_{counter}', []), 'color': (0,255,0)}], 'result')

        #cv2.imwrite(f"results/{args.alpha}_{args.percentage}/fg_{counter}.png", foreground)
        if not evaluate:
            writer.append_data(foreground)
        counter += 1

        ret = model.compute_next_foreground()
        if ret:
            foreground, I = ret
        else:
            foreground = None

        if False and not evaluate and counter % 100 == 0:
            print(f"{counter} frames processed...")

        if args.max != -1 and counter >= args.max:
            break

    mAP = utils.get_AP(gt_rects, det_rects)
    if not evaluate:
        print(f"DONE! {counter} frames processed")
        writer.close()
        print(f"Saved to '{results_path}'")
        print('mAP:', mAP)

    return mAP

def get_args():
    parser = argparse.ArgumentParser(description='Extract foreground from video.')
    parser.add_argument('-m', '--model', type=str, default='gm', choices=["gm", "agm"], help="model used for background modeling")
    parser.add_argument('-c', '--colorspace', type=str, default='gray', choices=["gray", "rgb", "hsv", "lab", "ycrcb"], help="colorspace used for background modeling")
    parser.add_argument('-M', '--max', type=int, default=-1, help="max of frames for which infer foreground")
    parser.add_argument('-perc', '--percentage', type=float, default=0.25, help="percentage of video to use for background modeling")
    parser.add_argument('-r', '--random', action='store_true', help="if set, random search will be used instead of grid search")
    parser.add_argument('-o', '--output', type=str, default='output.csv', help="output file to store the csv data")
    return parser.parse_args()

def run_search():
    args = get_args()
    combs = []
    if args.random:
        combs = [(np.floor(random.random() * 22 + 1), np.floor(random.random() * 200) / 1000) for i in range(1)]
    else:
        alphas = list(range(1, 23, 4))
        ps = [val / 1000 for val in list(range(0, 201, 25))]
        combs = [(alpha, p) for alpha in alphas for p in ps]

    total = len(combs)
    print(f"{'RANDOM' if args.random else 'GRID'} SEARCH")
    print(f"combs={combs}")
    print(f"TOTAL: {total}")
    results = []
    pbar = tqdm(total=total)
    for alpha, p in combs:
        args.alpha = [alpha, ]
        args.p = p
        try:
            mAP = main(args)
        except:
            mAP = -1
        results.append((args.model, args.alpha, args.p, mAP))

    df = pd.DataFrame(data=results, columns=["model", "alpha", "mAP"])
    df.to_csv(args.output, index=False)


run_search()