# Week1

## Tasks 1 and 2
Tasks 1.1, 1.2 and 2 are implemented between the files main.py and utils.py. 

Detection file, noise configuration and visualization options must be changed on main.py. Important "global" variables are:

VIDEO_PATH = *path to vdo.avi*
annotations_file = *path to annotations.xml with complete ground truth data (including still objects, bikes and cars)*

> This file can be parsed with the function utils.parse_xml_rects)

GT_PATH = *path to AICity ground truth data, which contains only moving objects*

> This file can be parsed with the function utils.parse_aicity_rects

Please, find python dependencies on week1/requirements.txt

RUN_NAME = *sub-folder name in which files (like plots, frames and pkl data) will be stored* 

> Note a directory called *runs* must be created on the running directory

DET_PATH = *path to AICity detection data*

> This file can be parsed with the function utils.parse_aicity_rects

For the following weeks, this configuration will hopefully be treated on a more robust/user-friendly manner

## Tasks 3 and 4
These tasks are implemented in their correspondent Jupyter Notebooks. More information inside