# MCV-M6-Video-Analysis

# Team 1

| Members | Contact |
| :---         |   :---    | 
| Aditya Rana   | adityasangramsingh.rana@e-campus.uab.cat | 
| German Barquero    | german.barquero@e-campus.uab.cat  |
| Carmen García    | carmen.garciano@e-campus.uab.cat  |
| Juan Chaves | juanvictor.chaves@e-campus.uab.cat |


# Week1

## Tasks 1 and 2
Tasks 1.1, 1.2 and 2 are implemented between the files main.py and utils.py. 

Detection file, noise configuration and visualization options are chosen by cmd command. Despite that, somo important "global" variables are:

VIDEO_PATH = *path to vdo.avi*
ANNOTATIONS_FILE = *path to annotations.xml with complete ground truth data (including still objects, bikes and cars)*

> This file can be parsed with the function utils.parse_xml_rects

Please, find python dependencies on week1/requirements.txt

DET_PATH = *path to AICity detection data*

> This file can be parsed with the function utils.parse_aicity_rects. Path to rcnn, yolo and ssd detections already on the script. Select using the option: mode.

### Command line call
usage: main.py [-h] -m MODE -n NAME [-d] [-s] [--noise NOISE]

optional arguments:

  -h, --help            show this help message 
  and exit

  -m MODE, --mode MODE  yolo, rcnn or ssd

  -n NAME, --name NAME  Storage older name

  -d, --display         Whether to display the 
  video or not

  -s, --save            Wheter to save frames 
  and graphics for each of them or not

  --noise NOISE         Noise addition configuration. Format drop-pos-size-ar


## Tasks 3 and 4
These tasks are implemented in their correspondent Jupyter Notebooks. More information inside