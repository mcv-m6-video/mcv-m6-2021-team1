
# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Compute metrics for trackers using MOTChallenge ground-truth data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import OrderedDict
import glob
import logging
import os
from pathlib import Path
import pandas as pd

import motmetrics as mm
from week5.utils import get_GT_path, get_TRACKING_path

VIDEOS_LIST = ((1, list(range(1,6))), (3, list(range(10,16))), (4, list(range(16,41))))

def parse_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="""
Compute metrics for trackers using MOTChallenge ground-truth data.

Files
-----
All file content, ground truth and test files, have to comply with the
format described in

Milan, Anton, et al.
"Mot16: A benchmark for multi-object tracking."
arXiv preprint arXiv:1603.00831 (2016).
https://motchallenge.net/

Structure
---------

Layout for ground truth data
    <GT_ROOT>/<SEQUENCE_1>/gt/gt.txt
    <GT_ROOT>/<SEQUENCE_2>/gt/gt.txt
    ...

Layout for test data
    <TEST_ROOT>/<SEQUENCE_1>.txt
    <TEST_ROOT>/<SEQUENCE_2>.txt
    ...

Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
string.""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-s', '--sequence', type=int, default=-1, help="sequence")
    parser.add_argument('-c', '--camera', type=int, default=-1, help="camera")
    parser.add_argument('-f', '--folder', type=str, default="../output_post", help="tracking folder")
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--fmt', type=str, help='Data format', default='mot15-2D')
    parser.add_argument('--solver', type=str, help='LAP solver to use for matching between frames.')
    parser.add_argument('--id_solver', type=str, help='LAP solver to use for ID metrics. Defaults to --solver.')
    parser.add_argument('--exclude_id', dest='exclude_id', default=False, action='store_true', help='Disable ID metrics')
    return parser.parse_args()


def compare_dataframes(gts, ts):
    """Builds accumulator for each sequence."""
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logging.info('Comparing %s...', k)
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logging.warning('No ground truth for %s, skipping.', k)

    return accs, names


def main(args, sequence, cameras):
    # pylint: disable=missing-function-docstring
    GT_PATHS = [get_GT_path(sequence, camera).replace("gt/gt.txt", "") for camera in cameras]
    TR_PATHS = [get_TRACKING_path(sequence, camera, folder=args.folder) for camera in cameras]

    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(args.loglevel))
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if args.solver:
        mm.lap.default_solver = args.solver

    gtfiles = [[glob.glob(os.path.join(GT_PATH, 'gt/gt.txt'))[0] for f in TR_PATH] for GT_PATH, TR_PATH in zip(GT_PATHS, TR_PATHS)]
    tsfiles = [[f for f in glob.glob(os.path.join(TR_PATH, '*.txt')) if not os.path.basename(f).startswith('eval')] for TR_PATH in TR_PATHS]
    gtfiles = list(zip(*[[glob.glob(os.path.join(GT_PATH, 'gt/gt.txt'))[0] for f in TR_PATH] for GT_PATH, TR_PATH in zip(GT_PATHS, TR_PATHS)]))
    tsfiles = list(zip(*[[f for f in glob.glob(os.path.join(TR_PATH, '*.txt')) if not os.path.basename(f).startswith('eval')] for TR_PATH in TR_PATHS]))
    print(gtfiles[1])
    print(tsfiles[1])

    logging.info('Found %d groundtruths and %d test files.', len(gtfiles), len(tsfiles))
    logging.info('Available LAP solvers %s', str(mm.lap.available_solvers))
    logging.info('Default LAP solver \'%s\'', mm.lap.default_solver)
    logging.info('Loading files.')

    def process_different_cameras_pandas(array):
        # we put a different initial frame number for each one (10.000 is high enough to avoid overlap)
        for i, a in enumerate(array):
            as_list = a.index.tolist()
            offset = i * 10000
            frame = [frame + offset for (frame, tr_id) in as_list]
            tr_ids = [tr_id for (frame, tr_id) in as_list]
            a["FrameId"] = frame
            a["Id"] = tr_ids
            array[i] = a.set_index(["FrameId", "Id"])
        return pd.concat(array)

    gt = OrderedDict([(os.path.splitext(Path(f[0]).parts[-1])[0], process_different_cameras_pandas([mm.io.loadtxt(os.path.join(GT_PATH, "gt/gt.txt"), fmt=args.fmt, min_confidence=1) for GT_PATH in GT_PATHS])) for f in tsfiles])
    ts = OrderedDict([(os.path.splitext(Path(f[0]).parts[-1])[0], process_different_cameras_pandas([mm.io.loadtxt(TR_PATH, fmt=args.fmt) for TR_PATH in f])) for f in tsfiles])

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)

    metrics = list(mm.metrics.motchallenge_metrics)
    if args.exclude_id:
        metrics = [x for x in metrics if not x.startswith('id')]

    logging.info('Running metrics')

    if args.id_solver:
        mm.lap.default_solver = args.id_solver
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logging.info('Completed')


if __name__ == '__main__':
    args = parse_args()
    sequence, camera = args.sequence, args.camera

    if args.sequence != -1 and args.camera != -1:
        main(args, sequence, camera)
    else:
        for (sequence, cameras) in VIDEOS_LIST:
            main(args, sequence, cameras)
