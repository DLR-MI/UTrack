import os
import logging
from pathlib import Path
import numpy as np
import copy
import motmetrics as mm
from io import StringIO
import pandas as pd
mm.lap.default_solver = 'lap'

logger = logging.getLogger("ultralytics")

def unzip_objs(objs):
    if len(objs) > 0:
        tlwhs, ids, scores = zip(*objs)
    else:
        tlwhs, ids, scores = [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)

    return tlwhs, ids, scores

METRICS = ('mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall')

class MOTMetrics(object):

    def __init__(self, data_dir, val_ann, results_dir):
        self.data_dir = data_dir
        self.val_ann = val_ann
        self.results_dir = results_dir

    def load_annotations(self, seq_name):
        gt_type = '_val_half' if self.val_ann == 'val_half.json' else ''
        gt_filename = os.path.join(self.data_dir, 'train', seq_name, 'gt', f'gt{gt_type}.txt')
        self.gt_frame_dict = MOTMetrics.read_mot_results(gt_filename, is_gt=True)
        self.gt_ignore_frame_dict = MOTMetrics.read_mot_results(gt_filename, is_ignore=True)

    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, rtn_events=False):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        # gts
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

        # ignore boxes
        ignore_objs = self.gt_ignore_frame_dict.get(frame_id, [])
        ignore_tlwhs = unzip_objs(ignore_objs)[0]

        # remove ignored results
        keep = np.ones(len(trk_tlwhs), dtype=bool)
        iou_distance = mm.distances.iou_matrix(ignore_tlwhs, trk_tlwhs, max_iou=0.5)
        if len(iou_distance) > 0:
            match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
            match_is, match_js = map(lambda a: np.asarray(a, dtype=int), [match_is, match_js])
            match_ious = iou_distance[match_is, match_js]

            match_js = np.asarray(match_js, dtype=int)
            match_js = match_js[np.logical_not(np.isnan(match_ious))]
            keep[match_js] = False
            trk_tlwhs = trk_tlwhs[keep]
            trk_ids = trk_ids[keep]

        # get distance matrix
        iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)

        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and hasattr(self.acc, 'last_mot_events'):
            events = self.acc.last_mot_events  # only supported by https://github.com/longcw/py-motmetrics
        else:
            events = None
        return events

    def eval_file(self, filename):
        self.reset_accumulator()

        result_frame_dict = MOTMetrics.read_mot_results(filename, is_gt=False)
        #frames = sorted(list(set(self.gt_frame_dict.keys()) | set(result_frame_dict.keys())))
        frames = sorted(list(set(result_frame_dict.keys())))
        for frame_id in frames:
            trk_objs = result_frame_dict.get(frame_id, [])
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            self.eval_frame(frame_id, trk_tlwhs, trk_ids, rtn_events=False)

        return self.acc

    def evaluate(self, output_filename=None, duration=None, fps=None):
        names, accs = [], []
        for filename in sorted(self.results_dir.glob('*.txt')):
            self.load_annotations(filename.stem)
            accs.append(self.eval_file(filename))
            names.append(filename.stem)
        
        summary = MOTMetrics.get_summary(accs, names, formatted=False)
        MOTMetrics.save_summary(
            summary, self.results_dir.parent / output_filename, duration, fps, formatted=False
        )

    @staticmethod
    def get_summary(accs, names, metrics=METRICS, formatted=False):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=names,
            generate_overall=True
        )
        summary_str = mm.io.render_summary(
            summary, 
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        logger.info(summary_str)
        out = summary if not formatted else summary_str
        return out

    @staticmethod
    def save_summary(summary, filename, duration, fps, formatted=False):
        if formatted:
            df = pd.read_csv(StringIO(summary))
            columns = list(df.columns)[0].strip().split()
            df = df.apply(lambda x: x.values[0].strip().split(), result_type="expand", axis=1)
        else:
            df = summary
            columns = list(map(lambda x: x.upper(), df.columns))
            df.reset_index(inplace=True)
            
        df.columns = ['VIDEO'] + columns    
        df['duration'] = [duration] * df.shape[0]
        df['FPS'] = [fps] * df.shape[0]
        df.to_csv(filename, index=False, float_format='%.3f')
    
    @staticmethod
    def read_mot_results(filename, is_gt=False, is_ignore=False):
        valid_labels = {1}
        ignore_labels = {2, 7, 8, 12}
        results_dict = dict()
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                for line in f.readlines():
                    linelist = line.split(',')
                    if len(linelist) < 7:
                        continue
                    fid = int(linelist[0])
                    if fid < 1:
                        continue
                    results_dict.setdefault(fid, list())

                    box_size = float(linelist[4]) * float(linelist[5])

                    if is_gt:
                        if 'MOT16-' in filename or 'MOT17-' in filename:
                            label = int(float(linelist[7]))
                            mark = int(float(linelist[6]))
                            if mark == 0 or label not in valid_labels:
                                continue
                        score = 1
                    elif is_ignore:
                        if 'MOT16-' in filename or 'MOT17-' in filename:
                            label = int(float(linelist[7]))
                            vis_ratio = float(linelist[8])
                            if label not in ignore_labels and vis_ratio >= 0:
                                continue
                        else:
                            continue
                        score = 1
                    else:
                        score = float(linelist[6])

                    #if box_size > 7000:
                    #if box_size <= 7000 or box_size >= 15000:
                    #if box_size < 15000:
                        #continue

                    tlwh = tuple(map(float, linelist[2:6]))
                    target_id = int(linelist[1])

                    results_dict[fid].append((tlwh, target_id, score))

        return results_dict

"""
TrackEval: eval.py
"""

import time
import traceback
from multiprocessing.pool import Pool
from functools import partial
import os
from . import utils
from .utils import TrackEvalException
from . import timer
from .collections.count import Count

try:
    import tqdm
    TQDM_IMPORTED = True
except ImportError as _:
    TQDM_IMPORTED = False


class TrackEvalMetrics(object):
    """Evaluator class for evaluating different metrics for different datasets"""

    @staticmethod
    def get_default_eval_config():
        """Returns the default config values for evaluation"""
        code_path = utils.get_code_path()
        default_config = {
            'USE_PARALLEL': False,
            'NUM_PARALLEL_CORES': 8,
            'BREAK_ON_ERROR': True,  # Raises exception and exits with error
            'RETURN_ON_ERROR': False,  # if not BREAK_ON_ERROR, then returns from function on error
            'LOG_ON_ERROR': os.path.join(code_path, 'error_log.txt'),  # if not None, save any errors into a log file.

            'PRINT_RESULTS': True,
            'PRINT_ONLY_COMBINED': False,
            'PRINT_CONFIG': True,
            'TIME_PROGRESS': True,
            'DISPLAY_LESS_PROGRESS': True,

            'OUTPUT_SUMMARY': True,
            'OUTPUT_EMPTY_CLASSES': True,  # If False, summary files are not output for classes with no detections
            'OUTPUT_DETAILED': True,
            'PLOT_CURVES': True,
        }
        return default_config

    def __init__(self, config=None):
        """Initialise the evaluator with a config file"""
        self.config = utils.init_config(config, self.get_default_eval_config(), 'Eval')
        # Only run timing analysis if not run in parallel.
        if self.config['TIME_PROGRESS'] and not self.config['USE_PARALLEL']:
            timer.DOtimer = True
            if self.config['DISPLAY_LESS_PROGRESS']:
                timer.DISPLAY_LESS_PROGRESS = True

    @timer.time
    def evaluate(self, dataset_list, metrics_list, show_progressbar=False):
        """Evaluate a set of metrics on a set of datasets"""
        config = self.config
        metrics_list = metrics_list + [Count()]  # Count metrics are always run
        metric_names = utils.validate_metrics_list(metrics_list)
        dataset_names = [dataset.get_name() for dataset in dataset_list]
        output_res = {}
        output_msg = {}

        for dataset, dataset_name in zip(dataset_list, dataset_names):
            # Get dataset info about what to evaluate
            output_res[dataset_name] = {}
            output_msg[dataset_name] = {}
            tracker_list, seq_list, class_list = dataset.get_eval_info()
            print('\nEvaluating %i tracker(s) on %i sequence(s) for %i class(es) on %s dataset using the following '
                  'metrics: %s\n' % (len(tracker_list), len(seq_list), len(class_list), dataset_name,
                                     ', '.join(metric_names)))

            # Evaluate each tracker
            for tracker in tracker_list:
                # if not config['BREAK_ON_ERROR'] then go to next tracker without breaking
                try:
                    # Evaluate each sequence in parallel or in series.
                    # returns a nested dict (res), indexed like: res[seq][class][metric_name][sub_metric field]
                    # e.g. res[seq_0001][pedestrian][hota][DetA]
                    print('\nEvaluating %s\n' % tracker)
                    time_start = time.time()
                    if config['USE_PARALLEL']:
                        if show_progressbar and TQDM_IMPORTED:
                            seq_list_sorted = sorted(seq_list)

                            with Pool(config['NUM_PARALLEL_CORES']) as pool, tqdm.tqdm(total=len(seq_list)) as pbar:
                                _eval_sequence = partial(eval_sequence, dataset=dataset, tracker=tracker,
                                                         class_list=class_list, metrics_list=metrics_list,
                                                         metric_names=metric_names)
                                results = []
                                for r in pool.imap(_eval_sequence, seq_list_sorted,
                                                   chunksize=20):
                                    results.append(r)
                                    pbar.update()
                                res = dict(zip(seq_list_sorted, results))

                        else:
                            with Pool(config['NUM_PARALLEL_CORES']) as pool:
                                _eval_sequence = partial(eval_sequence, dataset=dataset, tracker=tracker,
                                                         class_list=class_list, metrics_list=metrics_list,
                                                         metric_names=metric_names)
                                results = pool.map(_eval_sequence, seq_list)
                                res = dict(zip(seq_list, results))
                    else:
                        res = {}
                        if show_progressbar and TQDM_IMPORTED:
                            seq_list_sorted = sorted(seq_list)
                            for curr_seq in tqdm.tqdm(seq_list_sorted):
                                res[curr_seq] = eval_sequence(curr_seq, dataset, tracker, class_list, metrics_list,
                                                              metric_names)
                        else:
                            for curr_seq in sorted(seq_list):
                                res[curr_seq] = eval_sequence(curr_seq, dataset, tracker, class_list, metrics_list,
                                                              metric_names)

                    # Combine results over all sequences and then over all classes

                    # collecting combined cls keys (cls averaged, det averaged, super classes)
                    combined_cls_keys = []
                    res['COMBINED_SEQ'] = {}
                    # combine sequences for each class
                    for c_cls in class_list:
                        res['COMBINED_SEQ'][c_cls] = {}
                        for metric, metric_name in zip(metrics_list, metric_names):
                            curr_res = {seq_key: seq_value[c_cls][metric_name] for seq_key, seq_value in res.items() if
                                        seq_key != 'COMBINED_SEQ'}
                            res['COMBINED_SEQ'][c_cls][metric_name] = metric.combine_sequences(curr_res)
                    # combine classes
                    if dataset.should_classes_combine:
                        combined_cls_keys += ['cls_comb_cls_av', 'cls_comb_det_av', 'all']
                        res['COMBINED_SEQ']['cls_comb_cls_av'] = {}
                        res['COMBINED_SEQ']['cls_comb_det_av'] = {}
                        for metric, metric_name in zip(metrics_list, metric_names):
                            cls_res = {cls_key: cls_value[metric_name] for cls_key, cls_value in
                                       res['COMBINED_SEQ'].items() if cls_key not in combined_cls_keys}
                            res['COMBINED_SEQ']['cls_comb_cls_av'][metric_name] = \
                                metric.combine_classes_class_averaged(cls_res)
                            res['COMBINED_SEQ']['cls_comb_det_av'][metric_name] = \
                                metric.combine_classes_det_averaged(cls_res)
                    # combine classes to super classes
                    if dataset.use_super_categories:
                        for cat, sub_cats in dataset.super_categories.items():
                            combined_cls_keys.append(cat)
                            res['COMBINED_SEQ'][cat] = {}
                            for metric, metric_name in zip(metrics_list, metric_names):
                                cat_res = {cls_key: cls_value[metric_name] for cls_key, cls_value in
                                           res['COMBINED_SEQ'].items() if cls_key in sub_cats}
                                res['COMBINED_SEQ'][cat][metric_name] = metric.combine_classes_det_averaged(cat_res)

                    # Print and output results in various formats
                    if config['TIME_PROGRESS']:
                        print('\nAll sequences for %s finished in %.2f seconds' % (tracker, time.time() - time_start))
                    output_fol = dataset.get_output_fol(tracker)
                    tracker_display_name = dataset.get_display_name(tracker)
                    for c_cls in res['COMBINED_SEQ'].keys():  # class_list + combined classes if calculated
                        summaries = []
                        details = []
                        num_dets = res['COMBINED_SEQ'][c_cls]['Count']['Dets']
                        if config['OUTPUT_EMPTY_CLASSES'] or num_dets > 0:
                            for metric, metric_name in zip(metrics_list, metric_names):
                                # for combined classes there is no per sequence evaluation
                                if c_cls in combined_cls_keys:
                                    table_res = {'COMBINED_SEQ': res['COMBINED_SEQ'][c_cls][metric_name]}
                                else:
                                    table_res = {seq_key: seq_value[c_cls][metric_name] for seq_key, seq_value
                                                 in res.items()}

                                if config['PRINT_RESULTS'] and config['PRINT_ONLY_COMBINED']:
                                    dont_print = dataset.should_classes_combine and c_cls not in combined_cls_keys
                                    if not dont_print:
                                        metric.print_table({'COMBINED_SEQ': table_res['COMBINED_SEQ']},
                                                           tracker_display_name, c_cls)
                                elif config['PRINT_RESULTS']:
                                    metric.print_table(table_res, tracker_display_name, c_cls)
                                if config['OUTPUT_SUMMARY']:
                                    summaries.append(metric.summary_results(table_res))
                                if config['OUTPUT_DETAILED']:
                                    details.append(metric.detailed_results(table_res))
                                if config['PLOT_CURVES']:
                                    metric.plot_single_tracker_results(table_res, tracker_display_name, c_cls,
                                                                       output_fol)
                            if config['OUTPUT_SUMMARY']:
                                utils.write_summary_results(summaries, c_cls, output_fol)
                            if config['OUTPUT_DETAILED']:
                                utils.write_detailed_results(details, c_cls, output_fol)

                    # Output for returning from function
                    output_res[dataset_name][tracker] = res
                    output_msg[dataset_name][tracker] = 'Success'

                except Exception as err:
                    output_res[dataset_name][tracker] = None
                    if type(err) == TrackEvalException:
                        output_msg[dataset_name][tracker] = str(err)
                    else:
                        output_msg[dataset_name][tracker] = 'Unknown error occurred.'
                    print('Tracker %s was unable to be evaluated.' % tracker)
                    print(err)
                    traceback.print_exc()
                    if config['LOG_ON_ERROR'] is not None:
                        with open(config['LOG_ON_ERROR'], 'a') as f:
                            print(dataset_name, file=f)
                            print(tracker, file=f)
                            print(traceback.format_exc(), file=f)
                            print('\n\n\n', file=f)
                    if config['BREAK_ON_ERROR']:
                        raise err
                    elif config['RETURN_ON_ERROR']:
                        return output_res, output_msg

        return output_res, output_msg


@timer.time
def eval_sequence(seq, dataset, tracker, class_list, metrics_list, metric_names):
    """Function for evaluating a single sequence"""

    raw_data = dataset.get_raw_seq_data(tracker, seq)
    seq_res = {}
    for cls in class_list:
        seq_res[cls] = {}
        data = dataset.get_preprocessed_seq_data(raw_data, cls)
        for metric, met_name in zip(metrics_list, metric_names):
            seq_res[cls][met_name] = metric.eval_sequence(data)
    return seq_res
