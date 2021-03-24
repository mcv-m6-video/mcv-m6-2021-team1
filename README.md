# MCV-M6-Video-Analysis

# Team 1

| Members | Contact |
| :---         |   :---    | 
| Aditya Rana   | adityasangramsingh.rana@e-campus.uab.cat | 
| German Barquero    | german.barquero@e-campus.uab.cat  |
| Carmen García    | carmen.garciano@e-campus.uab.cat  |
| Juan Chaves | juanvictor.chaves@e-campus.uab.cat |

# <a name="w3"></a> Week 3

## Requirements
Apart from the packages on requirements.txt. you must follow the instructions on [this link](https://github.com/LucaCappelletti94/pygifsicle) for installing pygifsicle (used for reducing gif size).

tldr:

- on Linux (well, Ubuntu) run 
    ```
    sudo apt-get install gifsicle
    ```
- On windows you must look for an installer [here](https://eternallybored.org/misc/gifsicle/)
- On Mac no further action (apart from pip install) is required, so you just relax and see your investment on Apple payoff.

If you want to use the state-of-the-art trackers implemented as part of the task 2.2, you need to run

    ```
    git submodule update --init --recursive
    ```
    
And follow the [installation instructions](https://github.com/STVIR/pysot/blob/master/INSTALL.md) from the PySot submodule.

## Data format
This week we have embraced MOTS Challenge format as our *official*  file format. All object labelling and detection is stored on a txt with **a line per detection** with the following format: 

```
'frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'
```

Unknown data is represented with a -1. Example:

```
1,-1,882.263,92.697,55.811,47.536,0.906,-1,-1,-1
```

## Visualizer
We have created a script for showing the video with our results, be it detection or tracking. The input to it always is a txt file with the described format.

1. Add the path to your detection file to the list *detections* on *visualizer.py* with the following format:

```python
    detections = [
        {
            'name': 'gt',
            'full-name': 'Ground truth',
            'color': (0, 255, 0),
            'rects': utils.parse_xml_rects(GT_RECTS_PATH),
            'tracking': False
        },
        {
            'name': 'R101+IoU',
            'full-name': 'Retina Net R1o1 FPN 3x rp 128 + IoU tracking',
            'color': utils.get_random_col(),
            'rects': utils.parse_aicity_rects('./detections/retina101_track.txt', zero_index=0),
            'tracking': True
        },
        # ...
    ]
```
We already provide the base and tracked detections we used inside *week3/detections/* and *week3/trackings_iou/* respectively

> We assume the first detection on the list corresponds to ground truth data.

- name: Name displayed on the bounding box on the visualizer
- full-name: Name displayed on the visualizer legend and on the output AP file
- color: color of the bounding box on (R, G, B). We use a helper funciton for generating a random color, but you can specify it yourself. 
- rects: The actual detection read with our parser functions (available for full annotations xml and mots challeng txt format)
- tracking: whether we want the visualizer to choose the color of the boxes based on tracking information

> This last parameter overwrites the selected color

Make sure, also, that your detection's 'name' is on the list USE_DET

```python
# This list is intented to make filtering visualizations easier
USE_DET = ['gt', 'aigt', 'yolo', 'ssd', 'retina50', 'retina101', 'rcnn', 'R101+IoU']
```

2. Launch the script visualizer.py. It is convenient if you do so from inside week3.

3. The video will play, painting provided detections. The following keyboard commands are available:

- q: Quits the program
- p: Changes visualization speed between 0, 15, 30 and 100 FPS
- s: Saves an snapshot to your current directory with the name save_{frame_number}.jpg
- g: Toggles **gif recording**. Once you press g, the recording starts, until you press g again. You have to press g before the video ends, otherwise the gif won't ne generated. You can create multiple gifs, just be patient after you press q or the video ends while they are generated. Gifs are saved to *out_visualizer/{run}/gifs/*.

> You can check if you are recording or not (together with FPS information) on the bottom right corner of the video

> A new folder is generated inside *out_visualizer* on each execution of the program.

## Tracking
### 2.1 . IOU Tracking

IOU tracking is performed using the script *iou_tracker.py*. Input is a txt file following the described format representing a detection. An output following the same format, but now holding id information. Both input and output paths must be specified inside *iou_tracker.py*'s main.

### 2.2. Kalman tracking + state-of-the-art trackers
All trackers were implemented under the same architecture so they can be easily run and tested using the file:
````
$ python w3_run_kalman.py -h
usage    w3_run_kalman.py [-h] [-o OUTPUT] [-d DETECTIONS] [-t TRACKER]
                        [-th THRESHOLD] [-tl TRACKER_LIFE] [-M MAX]

optional arguments:optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        where results will be saved
  -d DETECTIONS, --detections DETECTIONS
                        detections used for tracking. Options: {retinanetpre, retinanet101pre, maskrcnnpre, ssdpre, yolopre}
  -t TRACKER, --tracker TRACKER
                        tracker used. Options: {"kalman", "kcf", "siamrpn_mobile", "siammask"}
  -th THRESHOLD, --threshold THRESHOLD
                        threshold used to filter detections
  -tl TRACKER_LIFE, --tracker_life TRACKER_LIFE
                        tracker life
  -M MAX, --max MAX     max number of frames to run the tracker (by default it runs all video).
                        Set to '-1' by default.

````

The txt file with the results will be stored for posterior evaluation. A video with the tracking visual results will also be generated.


### 2.3. IDF1 computation
IDF is computed using the script *weeek3/test_metric.py* as follows:

```
python3 test_metric.py <GT_FOLDER> <DET_FOLDER>
```
GT_FOLDER and DET_FOLDER hold txt files with ground truth and detection data respectively. They must have the following structure:

```
Layout for ground truth data
    <GT_FOLDER>/<SEQUENCE_1>/gt/gt.txt
    <GT_FOLDER>/<SEQUENCE_2>/gt/gt.txt
    ...

Layout for test data
    <DET_FOLDER>/<SEQUENCE_1>.txt
    <DET_FOLDER>/<SEQUENCE_2>.txt
    ...
```
Ground truth and detection is matched according to SEQUENCE_X. Results are displayed on the console.

# <a name="w1"></a> Week 2

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
  -M MAX, --max MAX     max number of frames for which to extract foreground. 
                        Set to '-1' by default, which means take all the frames available.
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


# <a name="w1"></a> Week1

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

