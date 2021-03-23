# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

OUTPUT_DIR = './m6-experiments'

models = ["COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", "COCO-Detection/retinanet_R_50_FPN_3x.yaml"]


ds_name = 'm6-aicity'

DS_PATH = '/home/group07/m6-frames'

for model in models:
    for batch in [128]:

        experiment_name = f'{ds_name}_{model[15:-5]}_rp{batch}'

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model))
        cfg.DATASETS.TRAIN = (ds_name,)
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch   # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
        # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

        cfg.OUTPUT_DIR = os.path.join(OUTPUT_DIR, experiment_name)
        
        # Inference should use the config with parameters that are used in training
        # cfg now already contains everything we've set previously. We changed it a little bit for inference:
        print('Evaluating...')
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
        predictor = DefaultPredictor(cfg)

        with open(experiment_name+'.txt', 'w') as fp:
            for i, im_name in enumerate(os.listdir(DS_PATH)):
                frame_num = int(im_name.split('_')[-1][:-4])
                print(im_name, im_name.split('_')[-1][:-4])
                im = cv2.imread(os.path.join(DS_PATH, im_name))
                if im is None:
                    print('Error reading image')
                    continue
                out = predictor(im)
                inst = out['instances']
                inst = inst[inst.pred_classes == 2]
                print(inst)
                for bbox, conf in zip(inst.pred_boxes, inst.scores):
                    bbox = bbox.to('cpu').numpy()
                    conf = conf.to('cpu').numpy()
                    x1, y1, x2, y2 = bbox
                    #   'frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'
                    line = f'{frame_num}, -1, {x1}, {y1}, {x2-x1}, {y2-y1}, {conf}, -1, -1, -1\n'
                    fp.write(line)                    
                if i == 3:
                    break
        print('Done')




