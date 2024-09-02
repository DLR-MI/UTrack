import logging
import os
import shutil

import numpy as np
import torch
from tqdm import tqdm

from collections import defaultdict
from tracker.update import Tracker
from .timer import Timer
from ..config.utils.thresholds import reassign_ucmc

logger = logging.getLogger("ultralytics")


def reassign_tracker_thresholds(video_name, args, orig_thresh, orig_fps, orig_track_buffer):
    """
    Reasign track threshold, frame rate and buffer size: for MOTXX datasets
    """
    # Check different FPS in the tables at https://motchallenge.net/data/MOT17/
    if ('MOT17-13' in video_name) or ('MOT17-14' in video_name):
        args.tracker.frame_rate = 25
        args.tracker.track_buffer = 25
    elif ('MOT17-05' in video_name) or ('MOT17-06' in video_name):
        args.tracker.frame_rate = 14
        args.tracker.track_buffer = 14
    else:
        args.tracker.frame_rate = orig_fps
        args.tracker.track_buffer = orig_track_buffer

    # Reasign high_score thresholds
    if video_name == 'MOT17-01-FRCNN':
        args.tracker.thresholds.scores.high = 0.65
    elif video_name == 'MOT17-06-FRCNN':
        args.tracker.thresholds.scores.high = 0.65
    elif video_name == 'MOT17-12-FRCNN':
        args.tracker.thresholds.scores.high = 0.7
    elif video_name == 'MOT17-14-FRCNN':
        args.tracker.thresholds.scores.high = 0.67
    elif video_name in ['MOT20-06', 'MOT20-08']:
        args.tracker.thresholds.scores.high = 0.3
    else:
        args.tracker.thresholds.scores.high = orig_thresh

    # Bring defaults from UCMCTrack
    if args.tracker.uncertain.kalman.on_ground:
        reassign_ucmc(args, video_name)


class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(self, args, dataloader):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
        """
        self.dataloader = dataloader
        self.args = args

    def evaluate(self, model, result_folder, dataset, interpolate=False, pix_perm=False):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        results = []
        timer_avgs, timer_calls = [], []
        video_names = defaultdict()

        timer = Timer()
        orig_thresh = 1.0 * self.args.tracker.thresholds.scores.high
        orig_fps = 1 * self.args.tracker.frame_rate
        orig_track_buffer = 1 * self.args.tracker.track_buffer

        for cur_iter, batch_data in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            with torch.no_grad():
                # init tracker
                frame_id = int(batch_data["frame_id"])
                video_id = batch_data["video_id"]
                try:
                    video_name = batch_data["file_name"].split('/')[-3]
                except IndexError:
                    video_name = batch_data["file_name"].split('/')[0]

                reassign_tracker_thresholds(
                    video_name,
                    self.args,
                    orig_thresh,
                    orig_fps,
                    orig_track_buffer
                )

                if video_name not in video_names:
                    video_names[video_id] = video_name

                if frame_id == 1:
                    # Instantiate tracker after reassigning thresholds
                    tracker = Tracker(self.args.tracker)
                    
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        MOTEvaluator.write_results(result_filename, results, is_kitti=self.args.exp=='kitti')
                        results = []

                    timer_avgs.append(timer.average_time)
                    timer_calls.append(timer.calls)
                    timer.clear()

                # run model
                timer.tic()
                img = batch_data['img']
                if pix_perm:
                    h, w, _ = img.shape
                    h1, h2 = np.random.randint(0, high=h, size=2)
                    w1, w2 = np.random.randint(0, high=w, size=2)
                    img[h1, w1, ...], img[h2, w2, ...] = img[h2, w2, ...], img[h1, w1, ...]
                outputs = model(img)

                # run tracking
            if outputs[0].boxes is not None:
                online_targets = tracker.update(outputs[0], dataset=dataset, seq_name=video_name)
                online_tlwhs = []
                online_tlbrs = []
                online_ids = []
                online_scores = []
                online_classes = []
                for t in online_targets:
                    tlwh = t.tlwh if not t.is_on_ground else t.from_ground_tlwh # FIXME: tlbr too
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.tracker.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_tlbrs.append(t.tlbr)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        online_classes.append(t.cls)
                # save results
                results.append(
                    (frame_id, online_tlwhs, online_tlbrs, online_ids, online_scores, online_classes)
                )
            timer.toc()

            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                MOTEvaluator.write_results(result_filename, results, is_kitti=self.args.exp=='kitti')
                timer_avgs.append(timer.average_time)
                timer_calls.append(timer.calls)

        # interpolate results if desired
        if interpolate:
            logger.info('Interpolating results ...')
            data_orig = result_folder.parent / f"{result_folder.name}_raw"
            if data_orig.exists():
                shutil.rmtree(str(data_orig))
            shutil.move(str(result_folder), str(data_orig))
            result_folder.mkdir(exist_ok=True)
            MOTEvaluator.dti(data_orig, str(result_folder), is_kitti=self.args.exp=='kitti')

        timer_avgs = np.asarray(timer_avgs)
        timer_calls = np.asarray(timer_calls)
        all_time = np.dot(timer_avgs, timer_calls)
        avg_time = all_time / np.sum(timer_calls)
        fps = 1.0 / avg_time
        logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, fps))

        # Restablish original thresholds
        self.args.tracker.thresholds.scores.high = orig_thresh
        self.args.tracker.frame_rate = orig_fps
        self.args.tracker.track_buffer = orig_track_buffer
        return all_time, fps

    @staticmethod
    def write_results(filename, results, is_kitti=False):
        if not is_kitti:
            save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
        else:
            save_format = '{frame},{id},{cls},-1,-1,-1,{x1},{y1},{x2},{y2},-1,-1,-1,-1,-1,-1,-1,{s}\n'
        with open(filename, 'w') as f:
            for frame_id, tlwhs, tlbrs, track_ids, scores, classes in results:
                for tlwh, tlbr, track_id, score, class_ in zip(tlwhs, tlbrs, track_ids, scores, classes):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    x1, y1, x2, y2 = tlbr
                    if not is_kitti:
                        line = save_format.format(
                            frame=frame_id,
                            id=track_id,
                            x1=round(x1, 1),
                            y1=round(y1, 1),
                            w=round(w, 1),
                            h=round(h, 1),
                            s=round(score, 2)
                        )
                    else:
                        line = save_format.format(
                            frame=frame_id,
                            id=track_id,
                            cls=class_,
                            x1=round(x1, 1),
                            y1=round(y1, 1),
                            x2=round(x2, 1),
                            y2=round(y2, 1),
                            s=round(score, 2)
                        )
                    f.write(line)
        logger.info('save results to {}'.format(filename))

    @staticmethod
    def write_interpolated_results(filename, results, is_kitti=False):
        if not is_kitti:
            save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
            detcols = slice(2,6)
            cls_map = {}
        else:
            save_format = '{frame},{id},{cls},-1,-1,-1,{x1},{y1},{x2},{y2},-1,-1,-1,-1,-1,-1,-1,{s}\n'
            detcols = slice(6,10)
            cls_map = {0: 'Pedestrian', 2:'Car'}
        with open(filename, 'w') as f:
            for i in range(results.shape[0]):
                frame_data = results[i]
                frame_id = int(frame_data[0])
                track_id = int(frame_data[1])
                x1, y1, w, h = frame_data[detcols]
                if not is_kitti:
                    line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h, s=-1)
                else:
                    cls = cls_map.get(int(frame_data[2]), -1) 
                    line = save_format.format(frame=frame_id, id=track_id, cls=cls, x1=x1, y1=y1, x2=w, y2=h, s=-1)
                f.write(line)

    @staticmethod
    def dti(txt_path, save_path, n_min=25, n_dti=20, is_kitti=False):
        detcols = slice(6,10,1) if is_kitti else slice(2,6,1)
        
        for seq_txt in sorted(txt_path.glob('*.txt')):
            seq_name = str(seq_txt).split('/')[-1]
            seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=',')
            min_id = int(np.min(seq_data[:, 1]))
            max_id = int(np.max(seq_data[:, 1]))
            seq_results = np.zeros((1, 18 if is_kitti else 10), dtype=np.float64)
            
            for track_id in range(min_id, max_id + 1):
                index = (seq_data[:, 1] == track_id)
                tracklet = seq_data[index]
                tracklet_dti = tracklet
                if tracklet.shape[0] == 0:
                    continue
                n_frame = tracklet.shape[0]
                n_conf = np.sum(tracklet[:, 17 if is_kitti else 6] > 0.5)
                if n_frame > n_min:
                    frames = tracklet[:, 0]
                    frames_dti = {}
                    for i in range(0, n_frame):
                        right_frame = frames[i]
                        if i > 0:
                            left_frame = frames[i - 1]
                        else:
                            left_frame = frames[i]
                        # disconnected track interpolation
                        if 1 < right_frame - left_frame < n_dti:
                            num_bi = int(right_frame - left_frame - 1)
                            right_bbox = tracklet[i, detcols]
                            left_bbox = tracklet[i - 1, detcols]
                            for j in range(1, num_bi + 1):
                                curr_frame = j + left_frame
                                curr_bbox = (curr_frame - left_frame) * (right_bbox - left_bbox) / \
                                            (right_frame - left_frame) + left_bbox
                                frames_dti[curr_frame] = (curr_bbox, tracklet[i, 2])
                    num_dti = len(frames_dti.keys())
                    if num_dti > 0:
                        data_dti = np.zeros((num_dti, 18 if is_kitti else 10), dtype=np.float64)
                        for n in range(num_dti):
                            data_dti[n, 0] = list(frames_dti.keys())[n]
                            data_dti[n, 1] = track_id
                            if is_kitti:
                                data_dti[n, 6:10], data_dti[n, 2] = frames_dti[list(frames_dti.keys())[n]]
                                data_dti[n, 3:6] =[-1,-1,-1]
                                data_dti[n, 10:] = [-1]*7 + [1]
                            else:
                                data_dti[n, 2:6], _ = frames_dti[list(frames_dti.keys())[n]]
                                data_dti[n, 6:] = [1, -1, -1, -1]
                        tracklet_dti = np.vstack((tracklet, data_dti))
                seq_results = np.vstack((seq_results, tracklet_dti))
            save_seq_txt = os.path.join(save_path, seq_name)
            seq_results = seq_results[1:]
            seq_results = seq_results[seq_results[:, 0].argsort()]
            MOTEvaluator.write_interpolated_results(save_seq_txt, seq_results, is_kitti)
