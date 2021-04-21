

def get_GT_path(sequence, camera):
    # sequence: sequence ID integer, camera: camera ID integer
    return f"../data/aic19-track1-mtmc-train/train/S{sequence:02d}/c{camera:03d}/gt/gt.txt"

def get_TRACKING_path(sequence, camera, folder="output_post"):
    # sequence: sequence ID integer, camera: camera ID integer
    return f"{folder}/S{sequence:02d}/C{camera:03d}/"

def get_VIDEO_path(sequence, camera):
    # sequence: sequence ID integer, camera: camera ID integer
    return f"../data/aic19-track1-mtmc-train/train/S{sequence:02d}/c{camera:03d}/vdo.avi"
    
def get_DET_path(sequence, camera, algorithm):
    # sequence: sequence ID integer, camera: camera ID integer, algorithm:={yolo3, ssd512, mask_rcnn}
    return f"../data/aic19-track1-mtmc-train/train/S{sequence:02d}/c{camera:03d}/det/det_{algorithm}.txt"
