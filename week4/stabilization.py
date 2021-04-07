import os
import cv2
import numpy as np
import block_match as bm

prev_tx_list = []
prev_ty_list = []
prev_r_list = []

def visualize_flow(flow):
    hsv = np.zeros_like(flow, dtype=np.float32)
    hsv[...,1] = 255

    mag, ang = cv2.cartToPolar(flow[:, :,0], flow[:, :,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

def translation_mat(tx, ty):
    mat = np.eye(3)
    mat[0,2] = tx
    mat[1,2] = ty

    return mat

def rotation_mat(a, center):
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    T = translation_mat(center[0], center[1])
    T_inv = translation_mat(-center[0], -center[1])

    return T@R@T_inv

def affine_mat(angle, tx, ty, center):
    R = rotation_mat(angle, center)
    T = translation_mat(tx, ty)
    mat = R@T

    return mat

def apply_flow(img, flow, kernel_type='gaussian', kernel_size=11, memory=20, use_angle=False, ):
    out = np.zeros_like(img)

    if use_angle:
        angles = 180*np.arctan2(flow[:,:,1], flow[:,:,0])/np.pi

        # angles = angles.astype(np.float32)
        # flow = cv2.medianBlur(flow, 5)
        # angles = cv2.GaussianBlur(angles, (11, 11), 0)
        # angles[np.where(abs(angles - np.std(angles)) > np.mean(angles))] = 0

        current_angle = np.average(angles, weights=None)
        prev_r_list.append(current_angle)
        if len(prev_r_list) > MEMORY:
            prev_r_list.pop(0)

        angle = -np.sum(prev_r_list)
    else:
        angle = 0


    # Construct affinity between frames

    # Smoothing
    flow = flow.astype(np.float32)
    if kernel_type == 'median':
        flow = cv2.medianBlur(flow, kernel_size)
    elif kernel_type == 'gaussian':
        flow = cv2.GaussianBlur(flow, (kernel_size, kernel_size), 0)
    else:
        print('Unknown kernel type')

    # Poling
    current_tx = np.average(flow[:,:,0], weights=None)
    current_ty = np.average(flow[:,:,1], weights=None)

    # Accumulate
    prev_tx_list.append(current_tx)
    prev_ty_list.append(current_ty)

    if len(prev_tx_list) > memory:
        prev_tx_list.pop(0)
    if len(prev_ty_list) > memory:
        prev_ty_list.pop(0)

    tx = -np.sum(prev_tx_list)
    ty = -np.sum(prev_ty_list)

    # Apply
    H = affine_mat(angle, tx, ty, (img.shape[1]//2, img.shape[0]//2))
    img = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
    return img

def paint_grid(im):
    center = (im.shape[1]//2, im.shape[0]//2)
    im = cv2.circle(im, center, 4, (0,0,255), -1)

    im = cv2.line(im, (0, im.shape[0]//3), (im.shape[1], im.shape[0]//3), (0,0,255))
    im = cv2.line(im, (0, 2*im.shape[0]//3), (im.shape[1], 2*im.shape[0]//3), (0,0,255))
    im = cv2.line(im, (im.shape[1]//3, 0), (im.shape[1]//3, im.shape[0]), (0,0,255))
    im = cv2.line(im, (2*im.shape[1]//3, 0), (2*im.shape[1]//3, im.shape[0]), (0,0,255))
    return im

def main(videoname, kernel_type, kernel_size, memory, use_angle):
    cap = cv2.VideoCapture(cv2.samples.findFile(f'../../{VIDEO}.avi'))
    ret, frame1 = cap.read()

    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

    in_frames = []
    out_frames = []
    frame_cont = 0
    print('Reading video...')

    while(True):
        frame_cont += 1
        if not(frame_cont % 15) and frame_cont:
            print('On frame number', frame_cont)

        ret, frame2 = cap.read()
        if not ret:
            break

        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = bm.get_optical_flow(prvs, next, 16, 32, comp='forward')

        bgr = visualize_flow(flow)
        out = apply_flow(frame2, flow, kernel_type, kernel_size, memory, use_angle)

        # Reference grid
        frame2 = paint_grid(frame2)
        out = paint_grid(out)

        # cv2.imshow('flow',bgr)
        if DISPLAY:
            cv2.imshow('input',frame2)
            cv2.imshow('out',out)

        in_frames.append(frame2)
        out_frames.append(out)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break
        elif k == ord('s'):
            cv2.imwrite(f'frames/{frame_cont}_in.png',frame2)
            cv2.imwrite(f'frames/{frame_cont}_flow.png',bgr)
        prvs = next

    # Save video
    print('Saving videos...')
    os.makedirs('output', exist_ok=True)
    outname = f'output/out{videoname}_mem{memory}_typ{kernel_type}_ker{kernel_size}_angle_{use_angle}.avi'

    print('Output...')
    output = cv2.VideoWriter(outname, cv2.VideoWriter_fourcc(*'XVID'), 30, out.shape[:2])
    for f in out_frames:
        output.write(f)
    output.release()

    print('Input...')
    output = cv2.VideoWriter(f'output/in_{VIDEO}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, out.shape[:2])
    for f in in_frames:
        output.write(f)
    output.release()

if __name__== '__main__':
    TYPE = 'gaussian'
    KERNEL_SIZE = 11
    VIDEO = 'pc'
    DISPLAY = True
    MEMORY = 20
    USE_ANGLE = False

    main(VIDEO, TYPE, KERNEL_SIZE, MEMORY, USE_ANGLE)
