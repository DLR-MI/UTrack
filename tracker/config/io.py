import os
import getpass
import json
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf


def set_results_folder(exp, association, keep_last=False, submit=False, seed=None):
    results_folder = Path("track_results") / exp
    all_runs = list(results_folder.glob(association + '*'))
    if len(all_runs) == 0:
        suffix = '_1' if seed is None else f'_{seed}'
    else:
        # Runs with the same tracker go to different folders
        run = all_runs[-1].stem.split('_')
        shift = 0 if keep_last else 1
        incremented_run = str(int(run[-1]) + shift) 
        suffix = '_' + (incremented_run if seed is None else str(seed))
            
    results_folder /= Path(association + suffix) / 'data'
    results_folder.mkdir(parents=True, exist_ok=True)
    
    if submit:
        (results_folder.parent / 'results_test').mkdir(parents=True, exist_ok=True)
    return results_folder, association + suffix


def merge_with_cli_args(cli_args):
    try:
        args = OmegaConf.create(json.loads(cli_args.config.strip('"')))
    except:
        args = OmegaConf.load(cli_args.config)
    args_ucmc = OmegaConf.load(Path(__file__).parent / 'utils/ucmc.yaml')
    args.tracker.update(args_ucmc, merge=True)
    args.update(vars(cli_args), merge=True)
    return args


def save_config_tracker_run(args, save_dir):
    config_run = OmegaConf.create({
        'msg': args.msg, 
        'tracker': args.tracker,
        'run': {
            'user': getpass.getuser(),
            'exp': args.exp,
            'dataset': args.data_root.split('/')[-1],
            'seed': args.seed,
            'node': os.uname().nodename.split('.')[0],
            'gpu_id': args.yolo.device
        }
    })
    OmegaConf.save(config=config_run, f=str(save_dir / 'run_args.yaml'))


def set_seqmap(data_root, track_results, split_to_eval):
    with open(f'{data_root}/annotations/{split_to_eval}.json','r') as f: 
        ann = json.load(f)
    if not Path(track_results).exists():
        track_results.mkdir(parents=True)
    np.savetxt(
        f'{str(track_results)}/seqmap.txt', 
        [d['file_name'] for d in ann['videos']], fmt='%s'
    )
    

def save_duration(results_dir, duration, fps):
    df = pd.DataFrame({'duration': duration, 'FPS':fps}, index=[0])
    df.to_csv(results_dir / 'duration.csv', index=False)


"""
TrackEval: run_mot_challenge.py config
Note: the original metrics folder is renamed as collections. Not all metrics imported.
"""

def track_eval_config_mot(data_dir, val_ann, experiment, **kwargs):
    from ..eval.metrics import TrackEvalMetrics
    from ..eval.datasets.mot_challenge_2d_box import MotChallenge2DBox
    from ..eval.datasets.kitti_2d_box import Kitti2DBox
    from ..eval.collections.hota import HOTA
    from ..eval.collections.clear import CLEAR
    from ..eval.collections.identity import Identity
    from ..eval.collections.vace import VACE
    
    default_eval_config = TrackEvalMetrics.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    if experiment == 'kitti':
        default_dataset_config = Kitti2DBox.get_default_dataset_config()
    else:
        default_dataset_config = MotChallenge2DBox.get_default_dataset_config()
    
    # Rewrite default dataset config
    gt_split, gt_type = val_ann.split('.')[0], ''
    if 'val_half' in val_ann and not experiment == 'kitti':
        gt_split, gt_type = 'train', '_val_half'
        
    track_results = Path(__file__).parents[2] / f'track_results/{experiment}'
    default_dataset_config['SKIP_SPLIT_FOL'] = True
    default_dataset_config['GT_FOLDER'] = f'{data_dir}/{gt_split}'
    default_dataset_config['GT_LOC_FORMAT'] = '{gt_folder}/{seq}/gt/' + f'gt{gt_type}.txt'
    default_dataset_config['TRACKERS_FOLDER'] = track_results
    default_dataset_config['SEQMAP_FILE'] = f'{track_results}/seqmap.txt'
    default_dataset_config['SPLIT_TO_EVAL'] = kwargs['split_to_eval']
    default_dataset_config['BENCHMARK'] = 'MOT17'
    
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    
    args = {k.upper():v for k, v in kwargs.items()}
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting]:
                    x = True
                elif not args[setting]:
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            elif setting == 'SEQ_INFO':
                x = dict(zip(args[setting], [None]*len(args[setting])))
            else:
                x = args[setting]
            config[setting] = x
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}
    
    # Run code
    if experiment == 'kitti':
        dataset_list = [Kitti2DBox(dataset_config)]
    else:
        dataset_list = [MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [HOTA, CLEAR, Identity, VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    
    return eval_config, dataset_list, metrics_list 


    

