import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from flowvis import flow_to_color
from utils import read_flow, plot_flow, get_metrics
from tqdm import tqdm

METRICS = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

def exhaustive_search(template:np.ndarray, target:np.ndarray, metric='cv2.TM_CCORR_NORMED'):
    """
    search at all possible positions in target
    """
    # evaluate the openCV metric
    metric = eval(metric)
    result = cv2.matchTemplate(template, target, metric)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if metric in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        pos = min_loc
    else:
        pos = max_loc
    return pos

def logarithmic_search(template:np.ndarray, target:np.ndarray, metric:str='cv2.TM_SQDIFF_NORMED'):
    step=8
    orig = ((target.shape[0]-template.shape[0])//2, (target.shape[1]-template.shape[1])//2)
    step = (min(step, orig[0]), min(step, orig[1]))
    while step[0] > 1 and step[1] > 1:
        min_dist = np.inf
        pos = orig
        for i in [orig[0]-step[0], orig[0], orig[0]+step[0]]:
            for j in [orig[1]-step[1], orig[1], orig[1]+step[1]]:
                
                def _distance(x1,x2):
                    return np.mean((x1 - x2) ** 2)
                
                dist = _distance(template, target[i:i + template.shape[0], j:j + template.shape[1]])
                if dist < min_dist:
                    pos = (i, j)
                    min_dist = dist
        orig = pos
        step = (step[0]//2, step[1]//2)
    return orig

def find_template(template:np.ndarray, target:np.ndarray, search='exhaustive', metric='cv2.TM_CCORR_NORMED'):

    SEARCH_ALGOS = {
    'exhaustive': exhaustive_search,
    'logarithmic': logarithmic_search
    }

    return SEARCH_ALGOS[search](template, target, metric)

def get_optical_flow(img1:np.ndarray, img2:np.ndarray, block_size = 16, search_area = 16, 
                    comp = 'forward', search = 'exhaustive', metric = 'cv2.TM_CCORR_NORMED'):
    
    """
    Args:
        block_size: size of window (in px) to split the image into
        search_area: search area (in px) of expected movement of block
        algo: algorithm to find the displacement of the block
        
    Returns the optical flow from img1 --> img2 or vice-versa depending on compensation selected
    """
    
    assert img1.shape == img2.shape, "Got different image sizes"
    h,w = img1.shape[:2]
    
    if comp == 'forward':
        pass
    elif comp == 'backward':
        img1, img2 = img2, img1
    else:
        print('check docs for available compensations')
        
    flow = np.zeros((h, w, 2), dtype=float)
    
    for i in tqdm(range(0, h-block_size, block_size)):
         for j in range(0, w - block_size, block_size):
                # get bbox of target where template will be searched
                top_left = (max(i-search_area, 0), max(j-search_area, 0))
                bottom_right = min(i+block_size+search_area, h), min(j+block_size+search_area, w)
                
                template = img1[i:i+block_size, j:j+block_size]
                target = img2[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]
                
                displacement = find_template(template, target, search, metric)
                
                v_flow = displacement[1] - (j-top_left[1]) 
                u_flow = displacement[0] - (i-top_left[0])
                
                flow[i:i+block_size, j:j+block_size] = [u_flow, v_flow]
    flow = np.dstack((flow[:,:,0], flow[:,:,1], np.ones_like(flow[:,:,0])))
    return flow

if __name__ == '__main__':
    path = "/home/adityassrana/MCV_UAB/m6-va/project/Data/kitti/optical_flow/"

    img1 = cv2.imread(os.path.join(path,'color','000045_10.png'), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(path,'color','000045_11.png'), cv2.IMREAD_GRAYSCALE)

    gt_flow = read_flow(os.path.join(path,'gt','000045_10.png'))
    pred_flow = read_flow(os.path.join(path,'results','LKflow_000045_10.png'))

    block_flow = get_optical_flow(img1, img2, 16, 32, comp='forward')

    mse,pepn = get_metrics(gt_flow, pred_flow)
    print(f"LK_flow metrics, MSE:{mse}, PEPN:{pepn}")

    mse,pepn = get_metrics(gt_flow, block_flow)
    print(f"Block_flow metrics, MSE:{mse}, PEPN:{pepn}")