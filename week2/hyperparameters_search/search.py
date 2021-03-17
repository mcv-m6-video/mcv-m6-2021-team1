

import sys
sys.path.append("..")
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
VIDEO_PATH = "../../../data/AICity_data/train/S03/c010/vdo.avi"
GT_RECTS_PATH = "../../../data/ai_challenge_s03_c010-full_annotation.xml"

def main(args):

    if not os.path.exists(VIDEO_PATH):
        print("Video does not exist.")
        return

    model_frames = int(args.percentage * TOTAL_FRAMES)

    imgs_gif_bf = [] #gif generation
    imgs_gif_af = []
    
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
    elif args.model == "sota":
        model = Sota(VIDEO_PATH, model_frames, args.method)
        MODEL_NAME = "SOTA" + args.method
        args.model=args.method
        results_path = f"results/{MODEL_NAME}/{args.colorspace}_{args.method}"
    else:
        raise Exception

    counter = model.model_background()

    if not os.path.exists(results_path):
        os.makedirs(results_path)
    writer = imageio.get_writer(f"{results_path}/video.mp4", fps=25)

    det_rects = {}
    gt_rects = utils.parse_xml_rects(GT_RECTS_PATH, True)
    gt_rects = {k:v for k,v in gt_rects.items() if int(k.split('_')[-1]) >= int(TOTAL_FRAMES*args.percentage)} # remove "training" frames

    foreground, I = model.compute_next_foreground()
    foreground, recs = detection.post_processing(foreground, display=args.display, method=args.model)
    det_rects[f'f_{counter}'] = recs
    writer.append_data(foreground)
    
    counter = int(TOTAL_FRAMES*args.percentage)
    det_rects = {}
    gt_rects = utils.parse_xml_rects(GT_RECTS_PATH, True)
    gt_rects = {k:v for k,v in gt_rects.items() if int(k.split('_')[-1]) >= int(TOTAL_FRAMES*args.percentage)} # remove "training" frames

    gt_rects_detformat = {f: [{'bbox': r, 'conf':1} for r in v] for f, v in gt_rects.items()}

    while foreground is not None:

        #if args.display:
        #    utils.imshow_rects(I, [{'rects': recs, 'color': (0,0,255)}, 
        #        {'rects': gt_rects_detformat.get(f'f_{counter}', []), 'color': (0,255,0)}], 'result')


        #cv2.imwrite(f"results/{args.alpha}_{args.percentage}/fg_{counter}.png", foreground)
        writer.append_data(foreground)
        counter += 1

        ret = model.compute_next_foreground()
        if ret:
            foreground, I = ret

            if False and counter % 2 == 0:
                imgs_gif_bf.append(cv2.resize(foreground, (852,480))) #for gif generation

            foreground, recs = detection.post_processing(foreground, display=args.display, method=args.model)

            if False and counter % 2 == 0:
                img_gif = utils.imshow_rects(I, [{'rects': recs, 'color': (0,0,255)}, 
                    {'rects': gt_rects_detformat.get(f'f_{counter}', []), 'color': (0,255,0)}], 'result', disp=False)
                imgs_gif_af.append(foreground)

            det_rects[f'f_{counter}'] = recs
        else:
            foreground = None

        if counter % 100 == 0:
            print(f"{counter} frames processed...")

        if args.max != -1 and counter >= args.max:
            break

    print(f"DONE! {counter} frames processed")
    writer.close()
    print(f"Saved to '{results_path}'")

    # Remove first frames
    # det_rects = utils.parse_aicity_rects("../../data/AICity_data/train/S03/c010/gt/gt.txt")
    mAP = utils.get_AP(gt_rects, det_rects)
    #print('mAP:', mAP)

    #imageio.mimsave(f'{results_path}/before.gif', imgs_gif_bf[:200])
    #imgs_gif_af = [cv2.cvtColor(f_gif, cv2.COLOR_BGR2RGB) for f_gif in imgs_gif_af]
    #imageio.mimsave(f'{results_path}/after.gif', imgs_gif_af[:200])

    return mAP

def get_args():
    parser = argparse.ArgumentParser(description='Extract foreground from video.')
    parser.add_argument('-m', '--model', type=str, default='gm', choices=["gm", "agm"], help="model used for background modeling")
    parser.add_argument('-c', '--colorspace', type=str, default='gray', choices=["gray", "rgb", "hsv", "lab", "ycrcb"], help="colorspace used for background modeling")
    parser.add_argument('-M', '--max', type=int, default=-1, help="max of frames for which infer foreground")
    parser.add_argument('-perc', '--percentage', type=float, default=0.25, help="percentage of video to use for background modeling")
    parser.add_argument('-r', '--random', action='store_true', help="if set, random search will be used instead of grid search")
    parser.add_argument('-o', '--output', type=str, default='output.csv', help="output file to store the csv data")
    parser.add_argument('-d', '--display', action='store_true', help="Display frames as they are processed or not")
    return parser.parse_args()

def run_search():
    args = get_args()
    combs = []
    if args.random:
        #combs = [(np.floor(random.random() * 13 + 1), np.floor(random.random() * 200) / 1000) for i in range(10)]
        combs = [(np.floor(random.random() * 4 + 5), np.floor(random.random() * 90) / 1000 + 0.01) for i in range(10)]
    else:
        alphas = list(range(1, 12, 2))
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

        print((args.model, args.alpha, args.p, mAP))
        results.append((args.model, args.alpha, args.p, mAP))

    df = pd.DataFrame(data=results, columns=["model", "alpha", "p", "mAP"])
    df.to_csv(args.output, index=False)


run_search()