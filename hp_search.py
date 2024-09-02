import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Thread

from tracker.eval.hp_search.generation import generate_config
from tracker.eval.hp_search.visualize import run_dashboard
import tracker.eval.compare as compare
from tracker.config import io
from tracker.associations.collections import ASSOCIATIONS
from track import run_track


def make_parser():
    parser = argparse.ArgumentParser("HP_search")
    mot_choices = ['mix_17', 'mix_20', 'ablation_17', 'ablation_20', 'dancetrack', 'kitti']
    parser.add_argument("--num_seeds", type=int, default=20, help="number of seeds to eval per exp")
    parser.add_argument("--exp", type=str, default='ablation_17', choices=mot_choices, help="experiment name")
    parser.add_argument("--data_root", type=str, default='/data/MOT17', help="path to data, having benchmark name at the end.")
    parser.add_argument("--split_to_eval", type=str, default='val_half', choices=['val_half', 'val', 'train'], help="split to evaluate using TrackEval")
    parser.add_argument("--association", type=str, default=None, choices=list(ASSOCIATIONS.keys()), help="association model")
    parser.add_argument("--results_hp", type=str, default='/data/track_results_hp', help="path to results of hp search")
    parser.add_argument("--template_hp", type=str, default='hp_compare_ab17_ck1', help="template for configuration")
    parser.add_argument("--host", type=str, default="localhost", help="host for visualization server")
    parser.add_argument("--port", type=int, default=8085, help="port for visualization server")
    parser.add_argument("--gpu_id", type=int, default=0, help="cuda device")
    parser.add_argument("--visualize", action='store_true', help="whether to visualize the metrics")
    return parser.parse_args()
    
    
if __name__ == '__main__':
    args = make_parser()
    
    io.set_seqmap( # for trackeval
        args.data_root, 
        Path('track_results') / args.exp, 
        args.split_to_eval
    )
    seed_init = 1
    
    if args.visualize: 
        p = Thread(target=run_dashboard, args=(args.results_hp, args.host, args.port))
        p.start()
    
    for config_yaml in generate_config(args.association, args.gpu_id, args.template_hp):
        tasks_to_run = set()
        with ProcessPoolExecutor(max_workers=args.num_seeds) as e:
            for seed in range(seed_init, args.num_seeds + seed_init):
                future = e.submit(
                    run_track, 
                    exp=args.exp, 
                    seed=seed, 
                    config=config_yaml, 
                    results_hp=args.results_hp,
                    is_hp=True
                )
                tasks_to_run.add(future)
            for future in as_completed(tasks_to_run):
                future.result()
                
        if args.association is None:
            # Compare multiple associations with same hyperparameters
            association = json.loads(config_yaml)['tracker']['association']
        else:
            # Compare single association with multiple hyperparameters
            association = args.association
        compare.trackers(args.exp, association, args.results_hp)
    
    if args.visualize:
        p.join()
    