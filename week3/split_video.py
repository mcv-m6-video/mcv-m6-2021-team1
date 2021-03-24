import json, random, os
import cv2
from tqdm import tqdm


VIDEO_PATH = 'vdo.avi'
os.makedirs('video_frames')

cap = cv2.VideoCapture(VIDEO_PATH)
for idx in tqdm(range(2141)):
    _, frame = cap.read()
    cv2.imwrite(os.path.join('video_frames',f'f_{idx}.jpg'), frame)
