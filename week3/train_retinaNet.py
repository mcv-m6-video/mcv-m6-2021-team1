#export

import json, random, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
from utils import parse_aicity_rects, parse_xml_rects, get_AP

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from functools import partial
from itertools import chain

DATA_PATH = "AICity_data/train/S03/c010/"
VIDEO_PATH = os.path.join(DATA_PATH,'vdo.avi')
GT_PATH = os.path.join(DATA_PATH, 'ai_challenge_s03_c010-full_annotation.xml')
FRAMES_PATH = os.path.join(DATA_PATH, 'video_frames')

def get_datasect_dicts(start=0, end=540):
    dataset_dicts = []
    gt_rects = parse_xml_rects(GT_PATH)
    
    for idx in range(start ,end):
        record = {}

        file_name = os.path.join(FRAMES_PATH,f'f_{idx}.jpg')
        height,width = 1080, 1920
        
        record["file_name"] = file_name
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
        annos = gt_rects[f'f_{idx}']
        objs = []
        for anno in annos:
            bbox = anno['bbox']
            obj = {"bbox": list(map(int, bbox)),
                  "bbox_mode": BoxMode.XYXY_ABS,
                  "category_id": 0,
                  }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

DatasetCatalog.register("aicity_train", partial(get_datasect_dicts, 1080, 1620))
MetadataCatalog.get("aicity_train").set(thing_classes=["car"])
aicity_metadata = MetadataCatalog.get("aicity_train")

dataset_dicts = get_datasect_dicts(0, 540)

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("aicity_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001 # pick a good LR

cfg.SOLVER.WARMUP_ITERS = 300
cfg.SOLVER.MAX_ITER = 600
cfg.SOLVER.STEPS = (350, 500) #decay learning rate
cfg.SOLVER.GAMMA = 0.1

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.OUTPUT_DIR = 'r_test_split_2'# zero-indexed

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load()
trainer.train()

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

def get_datasect_dicts_test(s1=0, s2=0, s3=0, s4=0):
    dataset_dicts = []
    gt_rects = parse_xml_rects(GT_PATH)
    
    range1 = range(s1,s2)
    range2 = range(s3,s4)
    full_range = list(chain(range1, range2))

    for idx in full_range:
        record = {}

        file_name = os.path.join(FRAMES_PATH,f'f_{idx}.jpg')
        height,width = 1080, 1920
        
        record["file_name"] = file_name
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
        annos = gt_rects[f'f_{idx}']
        objs = []
        for anno in annos:
            bbox = anno['bbox']
            obj = {"bbox": list(map(int, bbox)),
                  "bbox_mode": BoxMode.XYXY_ABS,
                  "category_id": 0,
                  }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


DatasetCatalog.register("aicity_test", partial(get_datasect_dicts_test, 0, 1080, 1620, 2141))
MetadataCatalog.get("aicity_test").set(thing_classes=["car"])

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("aicity_test", ["bbox"], False, output_dir="./routput2/")
val_loader = build_detection_test_loader(cfg, "aicity_test")
print(inference_on_dataset(trainer.model, val_loader, evaluator))