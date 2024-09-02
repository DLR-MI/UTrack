import argparse
import logging
import random
import warnings
from functools import partial
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf.listconfig import ListConfig
from ultralytics import YOLO

from tracker.associations.collections import ASSOCIATIONS
from tracker.config import io
from tracker.eval.dataloader import MOTDataset
from tracker.eval.evaluators import MOTEvaluator
from tracker.eval.metrics import TrackEvalMetrics
import tracker.eval.compare as compare

logger = logging.getLogger("ultralytics")


def make_parser():
    parser = argparse.ArgumentParser("Tracker")
    mot_choices = ['ablation_17', 'ablation_20', 'mix_17', 'mix_20', 'dancetrack', 'kitti'
    ]
    project_choices = [f'yolov8{x}-mix' for x in ('n', 's', 'm', 'l', 'x')]
    parser.add_argument("--project", type=str, default='yolov8l-mix', choices=project_choices,
                        help="yolov8-mix pre-trained model")
    parser.add_argument("--exp", type=str, default='ablation_17', choices=mot_choices, help="experiment name")
    parser.add_argument("--template_hp", type=str, default=None, help="template for configuration")
    parser.add_argument("--association", type=str, default='uk_botsort', choices=list(ASSOCIATIONS.keys()),
                        help="association model")
    parser.add_argument("--config", type=str, default='./tracker/config/track_ablation_17.yaml',
                        help="path to config .yaml for the experiment")
    parser.add_argument("--data_root", type=str, default='/data/MOT17',
                        help="path to data, having benchmark name at the end.")
    parser.add_argument("--split_to_eval", type=str, default='val_half', choices=['val_half', 'train', 'val', 'test'],
                        help="split to evaluate using TrackEval")
    parser.add_argument("--submit_mot", type=bool, default=False, help="evaluate on test set for MOT submission")
    parser.add_argument("--same_run", type=bool, default=False, help="no new folder for tracker results. Handy for debugging")
    parser.add_argument("--seed", type=int, default=None, help="eval seed")
    parser.add_argument("--pix_perm", type=bool, default=False, help="add randomness by randomly permuting two pixels in every frame")
    parser.add_argument("--gpu_id", type=int, default=3, help="cuda device")
    parser.add_argument("--results_hp", type=str, default='/data/track_results_hp', help="path to results of hyperparameter optimization")
    parser.add_argument("--visualize", action='store_true', help="whether to visualize the metrics")
    return parser.parse_args()


def run_evaluation(model, args, results_folder, submit=False, pix_perm=False):
    out_dir = results_folder.parent / 'results_test' if submit else results_folder
    ann_file = 'test.json' if submit else args.data.val_ann
    dataset = args.data_root.split('/')[-1]
    name = ann_file.split('.')[0]
    if '_half' in name:
        name = 'train'
    # Load evaluation dataset
    val_loader = MOTDataset(
        data_dir = args.data_root,
        json_file = ann_file, 
        name = name,
    )
    # Start evaluate
    logger.info('Evaluating model ...')
    evaluator = MOTEvaluator(args=args, dataloader=val_loader)
    duration, fps = evaluator.evaluate(model, out_dir, dataset, interpolate=True, pix_perm=pix_perm)
    return duration, fps


def compute_metrics(args, tracker_to_eval):
    # Compute metrics
    freeze_support()
    eval_config, dataset_list, metrics_list = io.track_eval_config_mot(
        data_dir=args.data_root,
        val_ann=args.data.val_ann,
        experiment=args.exp,
        benchmark=args.data_root.split('/')[-1],
        split_to_eval=args.split_to_eval,
        trackers_to_eval=[tracker_to_eval],
        metrics=['HOTA', 'CLEAR', 'Identity'],
        use_parallel=False,
        num_parallel_cores=8,
    )
    metrics = TrackEvalMetrics(eval_config)
    metrics.evaluate(dataset_list, metrics_list)


def run_track(exp=None, seed=None, config=None, results_hp=None, is_hp=False):
    # Prepare tracking config
    args = make_parser()
    cli_association = args.association
    
    if is_hp:
        args.exp = exp
        args.seed = seed
        args.config = config
        args.results_hp = results_hp
        args.same_run = False
        args.pix_perm = True
    
    args = io.merge_with_cli_args(args)
    if cli_association is not None and not is_hp:
        # Override association by that provided in cmd line
        args.tracker.association = cli_association
    
    # Set up output folders
    results_folder, tracker_to_eval = io.set_results_folder(
        args.exp, args.tracker.association, args.same_run, submit=args.submit_mot, seed=args.seed
    )
    if results_hp is None:
        io.set_seqmap(  # for trackeval
            args.data_root,
            Path('track_results') / args.exp,
            args.split_to_eval
        )
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # cudnn.deterministic = True
        # warnings.warn("You have chosen to seed testing. This will turn on the CUDNN deterministic setting")
        
    # Load model
    model = YOLO(Path(args.project) / args.exp / 'weights/best.pt').to(args.yolo.device)
    predict_args = {k: (tuple(v) if isinstance(v, ListConfig) else v) for k, v in args.yolo.items()}
    model.predict = partial(model.predict, **predict_args)

    # Load dataset and evaluate 
    duration, fps = run_evaluation(model, args, results_folder, pix_perm=args.pix_perm)
    io.save_duration(results_folder.parent, duration, fps)

    # Compute CLEAR and TrackEval metrics
    if not args.submit_mot:
        compute_metrics(args, tracker_to_eval)

    # Postprocess
    io.save_config_tracker_run(args, results_folder.parent)
    if not is_hp and args.visualize:
        # Ensure the dashboard is running and reading 'results_hp'
        compare.trackers(args.exp, args.association, args.results_hp)
    
    ''' Run test for submission to MOT challenge '''
    if args.submit_mot:
        duration, fps = run_evaluation(model, args, results_folder, submit=True)
        io.save_duration(results_folder.parent, duration, fps)


if __name__ == '__main__':
    run_track()
