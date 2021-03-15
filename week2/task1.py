
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# 2141 frames in total
TOTAL_FRAMES = 2141
PERCENTAGE = 0.05
ALPHA = 11
VIDEO_PATH = "../../AICity_data/train/S03/c010/vdo.avi"

class GaussianModel:

    def __init__(self, frames, width, height, alpha):
        self.alpha = alpha
        self.images = np.zeros((width, height, frames))
        self.mean = None
        self.std = None
        self.frames = frames
        self.width = width
        self.height = height
        self.counter = 0

    def __compute_parameters(self):
        print("Modeling the background...")
        self.mean = self.images.mean(axis=-1)
        self.std = self.images.std(axis=-1)
        #cv2.imwrite("mean.png", self.mean*255)
        print("Done!")

    def get_foreground(self, I):
        if self.mean is None or self.std is None:
            self.__compute_parameters()
        
        return (I - self.mean >= self.alpha * (self.std + 2)).astype(np.uint8)*255

    def add(self, I):
        self.images[:,:,self.counter % self.frames] = I
        self.counter += 1

def main():
    if not os.path.exists(VIDEO_PATH):
        print("Video does not exist.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)

    flag, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    model_frames = int(PERCENTAGE * TOTAL_FRAMES)
    print(frame.shape)
    model = GaussianModel(model_frames, frame.shape[0], frame.shape[1], ALPHA)
    counter = 0
    while frame is not None:
        flag, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if counter < PERCENTAGE * TOTAL_FRAMES:
            model.add(frame)
            
        foreground = model.get_foreground(frame)
        #cv2.imwrite(f"results/fg_{counter}.png", foreground)
            
        #if counter > PERCENTAGE * TOTAL_FRAMES + 50:
        #    break

        counter += 1
    print(counter)

    cap.release()

main()