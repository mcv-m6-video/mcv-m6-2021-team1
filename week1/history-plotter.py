import pickle as pkl
from matplotlib import pyplot as plt

data = [
    {
'RUN_NAME' : 'rcnn-low-pa',
'DET_PATH' : '../../data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt'
    },
    {
'RUN_NAME' : 'ssd-low-pa',
'DET_PATH' : '../../data/AICity_data/train/S03/c010/det/det_ssd512.txt'
    },
    {
'RUN_NAME' : 'yolo-low-pa',
'DET_PATH' : '../../data/AICity_data/train/S03/c010/det/det_yolo3.txt'
    }
]

plt.figure()

for run in data:
    with open(f'runs/{run["RUN_NAME"]}_map_raw.pkl', 'rb') as f:
        miou_over_time = pkl.load(f)

    plt.plot(miou_over_time)

plt.legend([d['RUN_NAME'] for d in data])
plt.xlim([0, 2140])
plt.ylim([0, 1])
plt.xlabel('# Frame')
plt.ylabel('mAP')
plt.title(f'Mean Average Precision over time for all detections')
plt.savefig(f'runs/all_map.png')
plt.show()

