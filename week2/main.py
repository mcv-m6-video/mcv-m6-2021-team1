
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import imageio
import argparse
from models import GaussianModel, AdaptiveGaussianModel

import utils

# 2141 frames in total
TOTAL_FRAMES = 2141
#PERCENTAGE = 0.25
#ALPHA = 11
#P = 0.001

# VIDEO_PATH = "../../AICity_data/train/S03/c010/vdo.avi"
VIDEO_PATH = "../../data/AICity_data/train/S03/c010/vdo.avi"
GT_RECTS_PATH = "../../data/ai_challenge_s03_c010-full_annotation.xml"

def display_resized(name, im, sf=0.5):
    im = cv2.resize(im, tuple(np.int0(sf*np.array(im.shape[:2][::-1]))))
    cv2.imshow(name, im)
    k = cv2.waitKey(15)
    if k == ord('q'):
        quit()

def analyse_contours(im):
    det_recs = []

    out_im = np.zeros_like(im)

    im_c = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(im.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    im_area = im.shape[0]*im.shape[1]
    for i, cnt in enumerate(contours):
        # bbox
        x, y, w, h = cv2.boundingRect(cnt)

        # Metrics
        area = h*w
        area_pct = 100 * area / im_area
        length = cv2.arcLength(cnt, True)
        para = length/(area + 1e-6)
        compactness = 4*np.pi/area/length/length
        ar = h/w

        window = im[y:y+h, x:x+w]
        filling_factor = np.count_nonzero((window > 0)) / w / h

        # print(f'Area pct {area_pct}, para {para}, compact {compactness}, ar {ar}')

        # Filter
        # I'm just testing out values on the extracted features
        if area_pct < 0.5 or ar > 1.5 or ar < 0.5 or compactness > 5 or filling_factor < 0.3: 
            # Bad detection
            im = cv2.rectangle(im_c, (x, y), (x+w, y+h), (0, 0, 255), 3)
        else:
            # Good detection
            det_recs.append({'bbox': [x, y, x+w, y+h], 'conf': 1})

            out_im = cv2.drawContours(out_im, contours, i, (255,255,255), -1)
            im = cv2.rectangle(im_c, (x, y), (x+w, y+h), (0, 255, 0), 3)

            # print(f'Area pct {area_pct}, para {para}, compact {compactness}, ar {ar}, filling factor {filling_factor}')

    # display_resized('rects', im)
    return out_im, det_recs

def post_processing(foreground):
    # display_resized('before', foreground)

    # Filter out noise
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5)))

    # Filter using color
    # Random idea, okay-ish results, just ignore it
    if len(foreground.shape) > 2 and foreground.shape[-1] > 1:
        ch_mean = np.mean(foreground, axis=2)
        ch_std = np.std(foreground, axis=2)

        foreground[ch_mean < 200] = 0
        foreground[ch_std > 100] = 0
    
    # Get contours
    out_im, recs = analyse_contours(foreground)

    # TODO: NMS
    
    # display_resized('after', out_im)
    return out_im, recs


def main(args):

    if not os.path.exists(VIDEO_PATH):
        print("Video does not exist.")
        return

    model_frames = int(args.percentage * TOTAL_FRAMES)

    if args.model == "gm":
        model = GaussianModel(VIDEO_PATH, model_frames, args.alpha, \
                                checkpoint=f"{args.colorspace}_{args.percentage}", colorspace=args.colorspace)
        MODEL_NAME = "GaussianModel"
    elif args.model == "agm":
        model = AdaptiveGaussianModel(VIDEO_PATH, model_frames, args.alpha, args.p, \
                                checkpoint=f"{args.colorspace}_{args.percentage}", colorspace=args.colorspace)
        MODEL_NAME = "AdaptiveGaussianModel"
    else:
        raise Exception

    model.model_background()

    results_path = f"results/{MODEL_NAME}/{args.colorspace}_{args.alpha}_{args.percentage}"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    writer = imageio.get_writer(f"{results_path}/video.mp4", fps=25)

    foreground = model.compute_next_foreground()
    writer.append_data(foreground)
    counter = 0
    det_rects = {}
    while foreground is not None:
        foreground, recs = post_processing(foreground)
        det_rects[f'f_{counter}'] = recs
        #cv2.imwrite(f"results/{args.alpha}_{args.percentage}/fg_{counter}.png", foreground)
        writer.append_data(foreground)
        counter += 1

        foreground = model.compute_next_foreground()
        if counter % 100 == 0:
            print(f"{counter} frames processed...")

        if args.max != -1 and counter >= args.max:
            break

    print(f"DONE! {counter} frames processed")
    writer.close()
    print(f"Saved to '{results_path}'")

    gt_rects = utils.parse_xml_rects(GT_RECTS_PATH)
    mAP = utils.get_AP(gt_rects, det_rects)
    print('mAP:', mAP)

parser = argparse.ArgumentParser(description='Extract foreground from video.')
parser.add_argument('-m', '--model', type=str, default='gm', choices=["gm", "agm"], help="model used for background modeling")
parser.add_argument('-c', '--colorspace', type=str, default='gray', choices=["gray", "rgb", "hsv"], help="colorspace used for background modeling")
parser.add_argument('-M', '--max', type=int, default=-1, help="max of frames for which infer foreground")
parser.add_argument('-perc', '--percentage', type=float, default=0.25, help="percentage of video to use for background modeling")
parser.add_argument('-a', '--alpha', metavar='N', nargs='+', type=float, default=11, help="alpha value")
parser.add_argument('-p', '--p', type=float, default=0.001, help="[AdaptiveGaussianModel] parameter controlling the inclusion of new information to model")
args = parser.parse_args()

main(args)