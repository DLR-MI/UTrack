import subprocess
import json
from hashlib import sha256
from pathlib import Path

from omegaconf import OmegaConf
import pandas as pd
import argparse
pd.set_option("display.precision", 3)


METRICS = ['HOTA', 'MOTA', 'IDF1', 'DetA', 'AssA', 'FPS']
track_results = Path(__file__).parents[2] / 'track_results'


def trackers(exp, association='byte', results_hp=''):
    metrics_in_run = {}
    results_dir = track_results / exp

    df_out = pd.DataFrame(columns=['file', 'id', 'dataset', 'tracker'] + METRICS)
    for tracker_run in results_dir.glob('*'):
        current_association = '_'.join(tracker_run.name.split('_')[:-1])
        if (
            'csv' in tracker_run.name 
            or 'json' in tracker_run.name 
            or 'txt'  in tracker_run.name 
            or association != current_association
        ): 
            continue 
        dfm = pd.read_csv(tracker_run / 'duration.csv')
        dfe = pd.read_csv(tracker_run / 'pedestrian_summary.txt', sep=' ')
        df = pd.concat((dfe, dfm), axis=1)[METRICS]
        # Create id from tracker config and run args
        args_tracker = OmegaConf.load(tracker_run / 'run_args.yaml')
        str_tracker_config = json.dumps(OmegaConf.to_container(args_tracker.tracker))
        str_tracker_run = json.dumps(OmegaConf.to_container(args_tracker.run))
        id_run = sha256((str_tracker_run + str_tracker_config).encode()).hexdigest()
        bs = 'bs_' if len(current_association.split('_')) == 1 else ''
        df.insert(0, 'id', id_run[:7])
        df.insert(0, 'dataset', args_tracker.run.dataset)
        df.insert(0, 'tracker', bs + tracker_run.name)
        df_out = pd.concat((df_out, df), axis=0)
        metrics_in_run.update({id_run: OmegaConf.to_container(args_tracker)})
        
    # Store mean values for HOTA
    hota_mean, hota_std = df_out.HOTA.mean(), df_out.HOTA.std()  
    df_out['HOTA_mean'] = hota_mean
    df_out['HOTA_std'] = hota_std      
        
    # Sort according to HOTA
    df_out = df_out.sort_values(by='HOTA', ascending=False).reset_index(drop=True)        
    filename = sha256(''.join(list(metrics_in_run.keys())).encode()).hexdigest()[:7]
    filename += f'_{association}'
    df_out['file'] = filename
    print(f'Saving results to {str(results_dir)}/{filename}.csv')
    print(f'Summary: HOTA: {hota_mean:.1f} +/- {hota_std:.1f} at FPS: {df_out.FPS.mean():.1f}')
    
    df_out.to_csv(results_dir / f'{filename}.csv', index=False, float_format='%.3f')
    with open(results_dir / f'{filename}.json', 'w') as f:
        json.dump(metrics_in_run, f)
        
    # If doing hp_search, send results to respective folder
    if results_hp:
        subprocess.run(
            f'mv {results_dir}/*.{{csv,json}} {results_hp};',
            executable='/bin/bash', shell=True, check=True
        )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='ablation_17_fix', help='root dir of tracker runs')
    parser.add_argument('--association', default='byte', help='association model')
    parser.add_argument('--results_hp', default='/data/track_results_hp', help='dir to save hp results')
    args = parser.parse_args()
    trackers(args.exp, args.association, args.results_hp)