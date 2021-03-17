# MCV-M6-Video-Analysis

# Team 1

| Members | Contact |
| :---         |   :---    | 
| Aditya Rana   | adityasangramsingh.rana@e-campus.uab.cat | 
| German Barquero    | german.barquero@e-campus.uab.cat  |
| Carmen García    | carmen.garciano@e-campus.uab.cat  |
| Juan Chaves | juanvictor.chaves@e-campus.uab.cat |

# Week 2

## Runner
All tasks were implemented in `main.py`. The algorithm will either pre-compute the background modelling or load it if it has already been computed before and saved in the `checkpoints` folder. The algorithm will output a .mp4 video file with the result and a gif of the first 200 frame for visualization purposes. The different algorithms can be selected by playing with the scripts parameters:

## Sript Usage

The models available include
- GaussianModel -> 'gm'
- AdaptiveGM -> 'agm'
- SOTA -> 'sota', and select which one to use with "--method" argument to the parser

````
$ python week2/main.py -h
usage: main.py [-h] [-m {gm,agm,sota}] [-c {gray,rgb,hsv,lab,ycrcb}] [-M MAX] [-perc PERCENTAGE] 
               [-a N [N ...]] [-p P] [-d] [-meth {mog,mog2,lsbp,gmg,cnt,gsoc,knn}]

Extract foreground from video.

optional arguments:
  -h, --help            show this help message and exit
  -m {gm,agm,sota}, --model {gm,agm,sota}
                        The model used for background modeling. Default value is 'gm':Gaussian.
  -c {gray,rgb,hsv,lab,ycrcb}, --colorspace {gray,rgb,hsv,lab,ycrcb}
                        choose the colorspace used for background modeling. 
                        Default value is 'gray.
  -M MAX, --max MAX     max number of frames for which to extract foreground. Set to '-1' by default.
  -perc PERCENTAGE, --percentage PERCENTAGE
                        percentage of video to use for background modeling
  -a N [N ...], --alpha N [N ...]
                        alpha value or values depending on color space used for modelling
  -p P, --p P           Rho (p): [AdaptiveGaussianModel] parameter controlling the inclusion 
                        of new information to model
  -d, --display         to display frames as they are processed
  -meth {mog,mog2,lsbp,gmg,cnt,gsoc,knn}, --method {mog,mog2,lsbp,gmg,cnt,gsoc,knn}
                        SOTA algorithm used for background subtraction. 
                        The '--model' parameter has to be set to 'sota' to be able to use this.
````

## Random/Grid search
There is a folder specific for this with the hyperparameters search runner and the visualizer of the results (3D plot). We did not have time to implement an usable interface for this script and the parameters to try are hardcoded inside the script, as well as the main function, which was copied from the main runner.


# Week1

## Tasks 1 and 2
Tasks 1.1, 1.2 and 2 are implemented between the files `main.py` and `utils.py`. 

Detection file, noise configuration and visualization options are chosen by cmd command. Despite that, some important "global" variables are:

`VIDEO_PATH` = *path to vdo.avi*
`ANNOTATIONS_FILE` = *path to annotations.xml with complete ground truth data (including still objects, bikes and cars)*

This file can be parsed with the function utils.parse_xml_rects

`DET_PATH` = *path to AICity detection data*

This file can be parsed with the function utils.parse_aicity_rects. Path to rcnn, yolo and ssd detections already on the script. Select using the option: mode.

````
$ python main.py -h
usage: main.py [-h] -m MODE -n NAME [-d] [-s] [--noise NOISE]

optional arguments:
  -h, --help            show this help message and exit
  -m MODE, --mode MODE  yolo, rcnn or ssd
  -n NAME, --name NAME  Storage older name
  -d, --display         Whether to display the video or not
  -s, --save            Wheter to save frames and graphics for each of them or not
  --noise NOISE         Noise addition configuration. Format drop-pos-size-ar
````

## Tasks 3 and 4
The tasks are implemented in jupyter notebooks with their corresponding name.
