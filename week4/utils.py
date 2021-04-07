import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_flow(path:str):
    """
    Optical flow maps are saved as 3-channel uint16 PNG images: The first channel
    contains the u-component, the second channel the v-component and the third
    channel denotes if a valid ground truth optical flow value exists for that
    pixel (1 if true, 0 otherwise)
    """
    # cv2 flips the order of reading channels
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.double)
    
    # valid channel
    valid = img[:,:,0]
    
    # get flow vectors
    u_flow = (img[:,:,2] - 2**15)/64
    v_flow = (img[:,:,1] - 2**15)/64
    
    # remove invalid flow values
    u_flow[valid == 0] = 0
    v_flow[valid == 0] = 0
    
    # return image in correct order
    return np.dstack((u_flow, v_flow, valid))

def plot_flow(img):
    """
    plot u and v flows along with valid pixels
    """
    fig, axes = plt.subplots(1,3, figsize=(16,8))
    images = [img[:,:,0], img[:,:,1], img[:,:,2]]
    titles = ['u_flow','v_flow','valid']
    
    for ax,image,title in zip(axes.flatten(), images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=12)
    plt.tight_layout()
    #plt.savefig("figures_german/affinity.pdf")
    plt.show()
    
def get_metrics(gt_flow:np.ndarray, pred_flow:np.ndarray, mask:np.ndarray=None, th:int=3):
    """
    Calculate metrics from ground truth and predicted optical flow.
    The mask is usually the third channel of gt_flow
    
    Arguments:
        gt_flow,pred_flow are (H,W,3)
        th: motion_vector error greater than threshold is an erroneous pixel
        
    Returns:
     1. Mean Square Error in Non-occluded areas
     2. Percentage of Erroneous Pixels in Non-occluded areas
    """
    if mask is None:
        mask = gt_flow[:,:,2]  
    
    error = np.sqrt(np.sum((gt_flow[:,:,:2] - pred_flow[:,:,:2])**2, axis=-1))    
    msen = np.mean(error[mask != 0])
    pepn = 100 * np.sum(error[mask != 0] > th) / (mask != 0).sum()
    return msen, pepn

def show_field(flow, gray, step=30, scale=0.5):
    
    gray = np.copy(gray)
    plt.figure(figsize=(16,8))
    plt.imshow(gray, cmap='gray')
    
    U = flow[:, :, 0]
    V = flow[:, :, 1]
    H = np.hypot(U, V)

    (h, w) = flow.shape[0:2]
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))

    x = x[::step, ::step]
    y = y[::step, ::step]
    U = U[::step, ::step]
    V = V[::step, ::step]
    H = H[::step, ::step]

    plt.quiver(x, y, U, V, H, scale_units='xy', angles='xy', scale=scale)
    
    plt.axis('off')
    plt.show()