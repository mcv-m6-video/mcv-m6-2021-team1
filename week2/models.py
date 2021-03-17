
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
IDEAS:
    - Median
    - Remove outliers
"""

class Model:
    def __init__(self, video_path, num_frames, checkpoint, colorspace="gray"):
        self.cap = cv2.VideoCapture(video_path)
        self.images = None
        self.modeled = False
        self.num_frames = num_frames
        self.checkpoint = checkpoint
        self.channels = 3

        if colorspace == "gray":
            self.color_transform = cv2.COLOR_BGR2GRAY
            self.channels = 1
        elif colorspace == "hsv":
            self.color_transform = cv2.COLOR_BGR2HSV
        elif colorspace == "rgb":
            self.color_transform = cv2.COLOR_BGR2RGB
        else:
            raise Exception

    def __add_image(self, frame, pos):
        if len(frame.shape) == 2:
            if self.images is None:
                self.images = np.zeros((frame.shape[0], frame.shape[1], self.num_frames))
            self.images[:,:, pos] = frame
            return

        if self.images is None:
            self.images = np.zeros((frame.shape[0], frame.shape[1], self.channels, self.num_frames))
        self.images[:,:,:, pos] = frame

    def save_images(self):
        if self.images is not None:
            print("Background has already been modeled.")
            return

        flag, frame = self.cap.read()
        frame = cv2.cvtColor(frame, self.color_transform)
        self.__add_image(frame, 0)

        success, frame = self.cap.read()
        counter = 1

        print("Starting video processor to model background...")
        with tqdm(total=self.num_frames) as pbar:
            while success and frame is not None and counter < self.num_frames:
                frame = cv2.cvtColor(frame, self.color_transform)
                self.__add_image(frame, counter)
                counter += 1
                pbar.update(1)
                flag, frame = self.cap.read()

    def model_background(self):
        if self.checkpoint is not None and self.load_checkpoint() == 1:
            self.modeled = True
            frame = self.cap.read()
            counter = 1
            while frame is not None and counter < self.num_frames:
                frame = self.cap.read()
                counter += 1
            print("Background modeled!")
            return

        # we add all images to the array
        self.save_images()

        # we use the subclass specific method to model the background
        self.compute_parameters()
        self.modeled = True
        print("Background modeled!")

        if self.checkpoint is not None:
            self.save_checkpoint()
            print("Checkpoint saved!")


    def save_checkpoint(self):
        raise NotImplementedError("Must be overriden")
    
    def load_checkpoint(self):
        raise NotImplementedError("Must be overriden")

    def compute_parameters(self):
        raise NotImplementedError("Must be overriden")

class GaussianModel(Model):

    def __init__(self, video_path, num_frames, alpha, checkpoint=None, colorspace="gray"):
        super().__init__(video_path, num_frames, checkpoint, colorspace=colorspace)
        # 2 modes
        assert (colorspace != "gray" and len(alpha) == 3) or (colorspace == "gray" and len(alpha) == 1), f"Colorspace '{colorspace}' does not match number of alphas: {len(alpha)}"
        print(f"[INIT] GaussianModel - alpha={alpha} - colorspace={colorspace}")
        self.alpha = alpha
        self.mean = None
        self.std = None

        self.base = "./checkpoints/GaussianModel"

    def compute_parameters(self):
        """
            Function called after first X% of images are saved in self.images
            The values computed here will be used afterwards to compute the foreground
        """
        self.mean = self.images.mean(axis=-1)
        print("Mean computed successfully.")
        self.std = self.images.std(axis=-1)
        print("Standard deviation computed successfully.")

    def compute_next_foreground(self):
        """
            Function to compute the foreground. Values computed in function 'compute_parameters'
            are available to use.
        """
        if not self.modeled:
            print("[ERROR] Background has not been modeled yet.")
            return None
        
        success, I = self.cap.read()
        if not success:
            return None

        I = cv2.cvtColor(I, self.color_transform)
        return (abs(I - self.mean) >= self.alpha * (self.std + 2)).astype(np.uint8) * 255, I

    def save_checkpoint(self):
        """
            Save info of the modeled background
        """
        if not os.path.exists(f"{self.base}/{self.checkpoint}"):
            os.makedirs(f"{self.base}/{self.checkpoint}")

        np.save(f"{self.base}/{self.checkpoint}/mean.npy", self.mean)
        np.save(f"{self.base}/{self.checkpoint}/std.npy", self.std)
        cv2.imwrite(f"{self.base}/{self.checkpoint}/mean.png", self.mean)
        cv2.imwrite(f"{self.base}/{self.checkpoint}/std.png", self.std)
        
        assert (np.load(f"{self.base}/{self.checkpoint}/mean.npy") == self.mean).all()
        assert (np.load(f"{self.base}/{self.checkpoint}/std.npy") == self.std).all()
    
    def load_checkpoint(self):
        """
            Load info of the modeled background
        """
        mean_path = f"{self.base}/{self.checkpoint}/mean.npy"
        std_path = f"{self.base}/{self.checkpoint}/std.npy"
        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            return -1
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        print("Checkpoint loaded!")
        return 1



class AdaptiveGaussianModel(Model):

    def __init__(self, video_path, num_frames, alpha, p, checkpoint=None, colorspace="gray"):
        super().__init__(video_path, num_frames, checkpoint, colorspace=colorspace)
        # 2 modes
        assert (colorspace != "gray" and len(alpha) == 3) or (colorspace == "gray" and len(alpha) == 1), f"Colorspace '{colorspace}' does not match number of alphas: {len(alpha)}"
        print(f"[INIT] AdaptiveGaussianModel - alpha={alpha}, p={p} - colorspace={colorspace}")
        self.alpha = alpha
        self.p = p
        self.mean = None
        self.std = None

        self.base = "./checkpoints/GaussianModel"

    def compute_parameters(self):
        """
            Function called after first X% of images are saved in self.images
            The values computed here will be used afterwards to compute the foreground
        """
        self.mean = self.images.mean(axis=-1)#, dtype=np.float64)
        print("Mean computed successfully.")
        self.std = self.images.std(axis=-1)#, dtype=np.float64)
        print("Standard deviation computed successfully.")

    def compute_next_foreground(self):
        """
            Function to compute the foreground. Values computed in function 'compute_parameters'
            are available to use.
        """
        if not self.modeled:
            print("[ERROR] Background has not been modeled yet.")
            return None
        
        success, I = self.cap.read()
        if not success:
            return None
        I = cv2.cvtColor(I, self.color_transform)

        # ADAPTIVE STEP HERE
        bm = (I - self.mean >= self.alpha * (self.std + 2)) # background mask

        self.mean[bm] = (self.p * I[bm] + (1 - self.p) * self.mean[bm])
        aux = (I - self.mean)
        self.std[bm] = np.sqrt(self.p * aux[bm] * aux[bm] + (1 - self.p) * (self.std[bm] * self.std[bm]))

        return (I - self.mean >= self.alpha * (self.std + 2)).astype(np.uint8) * 255, I

    def save_checkpoint(self):
        """
            Save info of the modeled background
        """
        if not os.path.exists(f"{self.base}/{self.checkpoint}"):
            os.makedirs(f"{self.base}/{self.checkpoint}")

        np.save(f"{self.base}/{self.checkpoint}/mean.npy", self.mean)
        np.save(f"{self.base}/{self.checkpoint}/std.npy", self.std)
        cv2.imwrite(f"{self.base}/{self.checkpoint}/mean.png", self.mean)
        cv2.imwrite(f"{self.base}/{self.checkpoint}/std.png", self.std)
        
        assert (np.load(f"{self.base}/{self.checkpoint}/mean.npy") == self.mean).all()
        assert (np.load(f"{self.base}/{self.checkpoint}/std.npy") == self.std).all()
    
    def load_checkpoint(self):
        """
            Load info of the modeled background
        """
        mean_path = f"{self.base}/{self.checkpoint}/mean.npy"
        std_path = f"{self.base}/{self.checkpoint}/std.npy"
        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            return -1
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        print("Checkpoint loaded!")
        return 1

class Sota(Model):
    def __init__(self, video_path, num_frames, method, checkpoint=None, colorspace="gray"):
        super().__init__(video_path, num_frames, checkpoint, colorspace=colorspace)

        if method == "mog":
            self.method = cv2.bgsegm.createBackgroundSubtractorMOG(history=110, nmixtures=7)
        elif method == "mog2":
            self.method = cv2.createBackgroundSubtractorMOG2()
        elif method == "lsbp":
            self.method = cv2.bgsegm.createBackgroundSubtractorLSBP() 
        elif method == "gmg":
            self.method = cv2.bgsegm.createBackgroundSubtractorGMG()
        elif method == "cnt":
            self.method = cv2.bgsegm.createBackgroundSubtractorCNT()
        elif method == "gsoc":
            self.method = cv2.bgsegm.createBackgroundSubtractorGSOC()
        elif method == "knn":
            self.method = cv2.createBackgroundSubtractorKNN()
        else:
            raise Exception

    def compute_next_foreground(self):
        """
            Function to compute the foreground. 
        """
        success, frame = self.cap.read()
        if not success:
            return None
        fgmask = self.method.apply(frame)
        return fgmask, frame

    def model_background(self):
        frame = self.cap.read()
        counter = 1
        while frame is not None and counter < self.num_frames:
            frame = self.cap.read()
            counter += 1
        print("Background modeled!")
        return
