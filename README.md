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

### Model selection 
GaussianModel -> gm
AdaptiveGM -> agm
SOTA -> sota, and select which one with "--method" 
```
'-m', '--model', type=str, default='gm', choices=["gm", "agm", "sota"]
```

### Method selection (only if model==sota)
```
'-meth', '--method', type=str, default='mog', choices=["mog", "mog2", "lsbp", "gmg", "cnt", "gsoc", "knn"]
```

### Colorspace
```
'-c', '--colorspace', type=str, default='gray', choices=["gray", "rgb", "hsv", "lab", "ycrcb"]
```

### Alpha
```
'-a', '--alpha', metavar='N', nargs='+', type=float, default=11
```

### Rho (p)
```
'-p', '--p', type=float, default=0.001
```


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
