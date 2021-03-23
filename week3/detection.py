import cv2
import numpy as np

import utils

ROI_PATH = "../../Data/AICity_data/train/S03/c010/roi.jpg"

def post_processing(foreground, method, display=False):
    roi = cv2.imread(ROI_PATH, 0)/255

    if display:
        utils.display_resized('before', foreground)

    foreground = foreground * roi.astype(np.uint8)
    foreground = apply_morph(foreground, method)
    
    out_im, recs = analyse_contours_agm(foreground, display)


    # NMS
    recs = NMS(recs)

    # if display:
    #     utils.display_resized('after', out_im)
    return out_im, recs


def NMS(rects):
    """
    Performs non maximum suppression.
    Input is a list of rects in the format [{'bbox': [x1,y1,x2,y2], 'conf': 1}, ...]
    """
    idx = utils.non_max_suppression_fast(np.array([r['bbox'] for r in rects]), 0.5)
    return [r for i, r in enumerate(rects) if i in idx]

def apply_morph(im, method):
    if method == 'gm' or method == 'knn' or method=='lsbp' or method=='cnt':
         # Filter out noise    
        im = cv2.morphologyEx(im, cv2.MORPH_ERODE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        
        im = cv2.morphologyEx(im, cv2.MORPH_DILATE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
        
        im = cv2.morphologyEx(im, cv2.MORPH_ERODE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        
        im = cv2.morphologyEx(im, cv2.MORPH_DILATE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))

    elif method == 'gmg':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)

    elif method == 'mog' or method=='mog2':
         # Filter out noise    
        im = cv2.morphologyEx(im, cv2.MORPH_ERODE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        
        im = cv2.morphologyEx(im, cv2.MORPH_DILATE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
        
        im = cv2.morphologyEx(im, cv2.MORPH_ERODE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        
        im = cv2.morphologyEx(im, cv2.MORPH_DILATE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))

        im = cv2.morphologyEx(im, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))

    elif method == 'agm':
        im = cv2.morphologyEx(im, cv2.MORPH_DILATE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    else:
        im = cv2.morphologyEx(im, cv2.MORPH_DILATE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return im 


def analyse_contours_agm(im, display):
    
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
        para = length/(area + 1e-9)
        compactness = 4*np.pi*area/(length*length+1e-9)
        ar = h/w

        window = im[y:y+h, x:x+w]
        filling_factor = np.count_nonzero((window > 0)) / (w * h)

        # print(f'Area pct {area_pct}, para {para}, compact {compactness}, ar {ar}')

        # Filter
        # I'm just testing out values on the extracted features
        # if area_pct < 0.25 or area_pct > 20 or ar > 1.2 or ar < 0.3: 
        if w < 10 or h < 10 or ar > 1.5 or ar < 0.5 or area_pct < 0.1 or para > 0.5: 
            # Bad detection
            im = cv2.rectangle(im_c, (x, y), (x+w, y+h), (0, 0, 255), 3)
        else:
            # Good detection

            det_recs.append({'bbox': [x, y, x+w, y+h], 'conf': 1})

            out_im = cv2.drawContours(out_im, contours, i, (255,255,255), -1)
            im = cv2.rectangle(im_c, (x, y), (x+w, y+h), (0, 255, 0), 3)

            # print(f'Area pct {area_pct}, para {para}, compact {compactness}, ar {ar}, filling factor {filling_factor}')

    if display:
        utils.display_resized('rects', im)
    return out_im, det_recs