import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from flowvis import flow_to_color
from utils import read_flow, plot_flow
from utils import get_metrics, show_field
from tqdm import tqdm
from skimage.feature import match_template
from itertools import product
import time
import pandas as pd

def find_template(template, target):
    pos = (0,0)
    for i in range(target.shape[0]-template.shape[0]):
            for j in range(target.shape[1]-template.shape[1]):
                dist = distance(template, target[i:i+template.shape[0], j:j+template.shape[1]], metric)
                if dist < min_dist:
                    pos = (i, j)
                    min_dist = dist
    return pos

def visualize_block_matching_exhaustive(img1:np.ndarray, img2:np.ndarray, block_size = 16, search_area = 16, comp = 'forward', search = 'exhaustive'):


    
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
        
    for i in range(2*block_size, h-block_size, block_size):
        for j in range(2*block_size, w - block_size, block_size):
            # get bbox of target where template will be searched
            top_left = (max(i-search_area, 0), max(j-search_area, 0))
            bottom_right = min(i+block_size+search_area, h), min(j+block_size+search_area, w)

            template = img1[i:i+block_size, j:j+block_size]
            target = img2[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]
            pos = (0,0)

            # find block
            for m in range(0,target.shape[0]-template.shape[0],2):
                for n in range(0,target.shape[1]-template.shape[1],2):
                    #print(f'{target.shape=}')
                    #print(f'{template.shape=}')
                    pos = (m,n)

                    fig = plt.figure(figsize=(8, 6))
                    ax1 = plt.subplot(1, 2, 1)
                    ax2 = plt.subplot(1, 2, 2)

                    ax1.imshow(img1,cmap='gray')
                    ax1.set_axis_off()
                    ax1.set_title('img1')
                    rect_block_1 = plt.Rectangle((i, j), block_size, block_size, edgecolor='g', linewidth=3, facecolor='none')
                    ax1.add_patch(rect_block_1)

                    ax2.imshow(img2, cmap='gray')
                    ax2.set_axis_off()
                    ax2.set_title('img2')
                    rect_block2 = plt.Rectangle((top_left[0]+m,top_left[1]+n), block_size, block_size, 
                                          edgecolor='r', linewidth=3, facecolor='none')
                    
                    rect_search_area = plt.Rectangle((top_left[0], top_left[1]), bottom_right[0]-top_left[0], 
                                          bottom_right[1]-top_left[1], edgecolor='c', 
                                          linewidth=2, facecolor='none')
                    ax2.add_patch(rect_block2)
                    ax2.add_patch(rect_search_area)
                    plt.tight_layout()
                    plt.savefig(f'images/run10/temp_{i}_{j}_{m}_{n}.png', dpi=fig.dpi)
                    plt.close()
                    #plt.show()
                    #break
                #break
            break
        break

def visualize_block_matching_logarithmic(img1:np.ndarray, img2:np.ndarray, block_size = 16, search_area = 16, 
                             comp = 'forward', search = 'exhaustive'):
    
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
        
    for i in range(2*block_size, h-block_size, block_size):
        for j in range(2*block_size, w - block_size, block_size):
            # get bbox of target where template will be searched
            top_left = (max(i-search_area, 0), max(j-search_area, 0))
            bottom_right = min(i+block_size+search_area, h), min(j+block_size+search_area, w)

            template = img1[i:i+block_size, j:j+block_size]
            target = img2[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]
            
            step = 8
            orig = ((target.shape[0]-template.shape[0])//2, (target.shape[1]-template.shape[1])//2)
            print(f'{orig=}')
            step = (min(step, orig[0]), min(step, orig[1]))
            
            # find block
            counter = 0
            while step[0] > 1 and step[1] > 1:
                min_dist = np.inf
                pos = orig
                for m in [orig[0], orig[0]-step[0], orig[0]+step[0]]:
                    #print(f'{m=}')
                    for n in [orig[1], orig[1]-step[1], orig[1]+step[1]]:
                        #print(f'{n=}')
                        dist = distance(template, target[m:m + template.shape[0], n:n + template.shape[1]], 
                                        metric='mse')
                        if dist < min_dist:
                            print(f'{dist=}')
                            pos = (m, n)
                            min_dist = dist
                            print(f'{pos=}')
                            fig = plt.figure(figsize=(8, 6))
                            ax1 = plt.subplot(1, 2, 1)
                            ax2 = plt.subplot(1, 2, 2)

                            ax1.imshow(img1,cmap='gray')
                            ax1.set_axis_off()
                            ax1.set_title('img1')
                            rect_block_1 = plt.Rectangle((i, j), block_size, block_size, edgecolor='g', 
                                                         linewidth=3, facecolor='none')
                            ax1.add_patch(rect_block_1)

                            ax2.imshow(img2, cmap='gray')
                            ax2.set_axis_off()
                            ax2.set_title('img2')
                            rect_block2 = plt.Rectangle((top_left[0]+pos[0],top_left[1]+pos[1]), 
                                                        block_size, block_size, 
                                                        edgecolor='r', linewidth=3, facecolor='none')

                            rect_search_area = plt.Rectangle((top_left[0]+orig[0]-step[0], 
                                                              top_left[1]+orig[1]-step[1]), block_size+2*step[0], 
                                                              block_size+2*step[1], edgecolor='c', 
                                                              linewidth=2, facecolor='none')
                            ax2.add_patch(rect_search_area)
                            ax2.add_patch(rect_block2)
                            plt.tight_layout()
                            #plt.savefig(f'images/run9/temp_{i}_{j}_{counter}_{m}_{n}.png', dpi=fig.dpi)
                            #plt.close()
                            plt.show()
                            break
                orig = pos
                step = (step[0]//2, step[1]//2)
                counter +=1 
            break
        break

if __name__ == '__main__':
    path = "/home/adityassrana/MCV_UAB/m6-va/project/Data/kitti/optical_flow/"
    img1_full = cv2.imread(os.path.join(path,'color','000157_10.png'), cv2.IMREAD_GRAYSCALE)
    img2_full = cv2.imread(os.path.join(path,'color','000157_10.png'), cv2.IMREAD_GRAYSCALE)

    gt_flow = read_flow(os.path.join(path,'gt','000157_10.png'))
    pred_flow = read_flow(os.path.join(path,'results','LKflow_000157_10.png'))

    img1 = img1_full[100:250, 450:600]
    img2 = img2_full[100:250, 450:600]

    visualize_block_matching_exhaustive(img1,img2,16,8)
    visualize_block_matching_logarithmic(img1,img2,16)