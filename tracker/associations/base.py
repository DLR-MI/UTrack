from omegaconf.listconfig import ListConfig
from ..matching import *
from ..basetrack import TrackState


class TrackUpdate(object):
    
    def __init__(self):
        # Persistent state
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        # Reset Temporal state (self.tracks)
        self.reset()
    
    def reset(self):
        self.tracks = dict(
            tracked = [],
            activated = [],
            refind = [],
            lost = [],
            removed = [],
            unconfirmed = []
        )
    
    def confirm(self):
        for track in self.tracked_stracks:
            if not track.is_activated:
                track.mark_unconfirmed()
                self.tracks['unconfirmed'].append(track)
            else:
               self.tracks['tracked'].append(track)
        
    def destroy(self, tracks, how='remove'):
        for track in tracks:
            if how == 'loose' and not track.state == TrackState.Lost:
                track.mark_lost()
                self.tracks['lost'].append(track)  
            elif how == 'remove':
                track.mark_removed()
                self.tracks['removed'].append(track)  
    
    def create(self, u_detections, kalman, thresh, frame_id):
        """ Init new stracks """
        for track in u_detections:
            if track.score < thresh:
                continue
            track.activate(kalman, frame_id)
            self.tracks['activated'].append(track)
            
    def update(self, frame_id, max_time_lost):
        """ Update state"""
        # Remove the tracks that have exceeded the max_tim_lost
        for track in self.lost_stracks:
            if frame_id - track.end_frame > max_time_lost:
                track.mark_removed()
                self.tracks['removed'].append(track)
                
        # Update the persistent state
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = TrackUpdate.joint_stracks(self.tracked_stracks, self.tracks['activated'])
        self.tracked_stracks = TrackUpdate.joint_stracks(self.tracked_stracks, self.tracks['refind'])
        self.lost_stracks = TrackUpdate.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(self.tracks['lost'])
        self.lost_stracks = TrackUpdate.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(self.tracks['removed'])
        return [track for track in self.tracked_stracks if track.is_activated]

    @staticmethod
    def read_detections(detector_output, low_score):
        # current detections
        bboxes = detector_output.boxes.xyxy.cpu().numpy()# x1y1x2y2 
        scores = detector_output.boxes.conf.cpu().numpy()
        classes = detector_output.boxes.cls.int().cpu().numpy()
        var_bboxes = detector_output.var_boxes.cpu().numpy()
        # Remove bad detections
        above_low = scores > low_score
        bboxes = bboxes[above_low]
        scores = scores[above_low]
        classes = classes[above_low]
        var_bboxes = var_bboxes[above_low]
        
        img = detector_output.orig_img
        return bboxes, scores, classes, var_bboxes, img

    @staticmethod
    def detection2strack(
        STrack, dets, *other, uncertain=False, fps=None, dataset=None, seq=None, doppler2phase=None
    ):
        if len(dets) > 0:
            '''Detections'''
            to_tlwh = STrack.tlbr_to_tlwh
            detections = [
                STrack(
                    to_tlwh(tlbr), 
                    *args, 
                    fps=fps, 
                    uncertain=uncertain, 
                    dataset=dataset, 
                    seq=seq,
                    doppler2phase=doppler2phase
                )
                for (tlbr, *args) in zip(dets, *other)
            ]
        else:
            detections = []
        return detections

    @staticmethod
    def deep_levels(tlist, depths):
        if len(tlist) == 0:
            return tlist
        deep_vecs = np.asarray([t.deep_vec for t in tlist])
        idx_sorted_by_depth = np.argsort(deep_vecs[:,-1])
        deep_levels = np.array_split(idx_sorted_by_depth, depths)
        return [[tlist[i] for i in l] for l in deep_levels if 0 not in l.shape]

    @staticmethod
    def joint_stracks(tlista, tlistb):
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res
    
    @staticmethod
    def sub_stracks(tlista, tlistb):
        stracks = {}
        for t in tlista:
            stracks[t.track_id] = t
        for t in tlistb:
            tid = t.track_id
            if stracks.get(tid, 0):
                del stracks[tid]
        return list(stracks.values())
    

class Thresholds(object):
    def __init__(self, cmc_method, STrack, **kwargs):
        self.new_track = kwargs['scores']['new_track']
        self.confirm = kwargs['scores']['confirm']
        self.low_score = kwargs['scores']['low']
        self.high_score = kwargs['scores']['high']
        self.matching = kwargs['matching']
        self.bins = kwargs['scores']['bins']
        self.bins = list(self.bins) if isinstance(self.bins, ListConfig) else self.bins
        self.depths = kwargs['pseudo_depth']['depths']
        self.cmc_method = cmc_method
        self.is_deep = kwargs['pseudo_depth']['add']
        self.STrack = STrack
    
    def _extend_to_num_bins(self, iterable, num_bins):
        last = iterable[-1]
        diff = num_bins - len(iterable)
        iterable += [last] * diff
    
    def define_binning_strategy(self, var_scores):
        if not isinstance(self.bins, (int, list)):
            bins = var_scores
        else:
            bins = self.bins
        return bins

    def bin_detections(self, scores, bins):
        if len(scores) == 0:
            return [[]], [[]], [[]], [[]]
        
        # Extension of BYTE association, to be able to go from 2 to more.
        # Note: the equal-size binning is done on the interval [low_score, high_score]
        if isinstance(bins, int):
            bin_size = (self.high_score - self.low_score) / (bins - 1)
            current_high = self.high_score
            
            output_indices = [scores > self.high_score]
            
            while (
                current_high > self.low_score + bin_size
                or np.isclose(current_high, self.low_score + bin_size)
            ):
                in_bin = np.logical_and(
                    scores > current_high - bin_size, 
                    scores < current_high
                )
                if np.any(in_bin):
                    output_indices.append(in_bin)
                    
                current_high -= bin_size
        
        # This is behind the cascade association in the KCM-Track paper
        elif isinstance(bins, list):
            output_indices = []
            previous_high = 1
            
            for current_high in bins[::-1] + [self.low_score]:
                in_bin = (scores > current_high) & (scores < previous_high)
                
                if np.any(in_bin):
                    output_indices.append(in_bin)
                previous_high = current_high    
                                        
        # This is our custom binning strategy based on score uncertainties.
        # Overlapping scores (within error) are merged, non-overlapping define different bins
        elif isinstance(bins, np.ndarray): # FIXME: if kcm used, requires ordering as above
            # bins are var_bboxes: used to merge score intervals
            std_scores = np.sqrt(bins)
            score_ranges = np.stack((scores - std_scores, scores + std_scores), axis=1)
            as_left_edge_sort = score_ranges[:,0].argsort()
            score_ranges = score_ranges[as_left_edge_sort]
            current_indices = [as_left_edge_sort[0]] 
            
            output_indices = []

            merged_score_ranges = []
            start_prev, end_prev = score_ranges[0]
            for i in range(1, len(score_ranges)):
                start_now, end_now = score_ranges[i]
                if start_now <= end_prev: # overlapping intervals, adjust the 'end'
                    end_prev = max(end_now, end_prev)
                    current_indices.extend([as_left_edge_sort[i]])
                else: # non-overlapping interval, add the previous interval and reset
                    merged_score_ranges.append([start_prev, end_prev])
                    output_indices.append(current_indices)
                    current_indices = [as_left_edge_sort[i]]
                    start_prev = start_now
                    end_prev = end_now    
            
            merged_score_ranges.append([start_prev, end_prev])
            if len(current_indices) > 1:
                output_indices.append(current_indices)  
            
            # if only one resulting bin, fall back to byte
            if len(output_indices) in (0, 1):
                return self.bin_detections(scores, bins=2)
            
            # if using kcm with auto binning
            if self.cmc_method == 'kcm':
                score_list = [end for _, end in merged_score_ranges]
                return self.bin_detections(scores, bins=score_list[:-1])
            
            output_indices = output_indices[::-1] # Reverse: we started with low scores
        
        # Extend provided per-level thresholds to the resulting number of bins
        num_bins = len(output_indices)
        self._extend_to_num_bins(self.matching['distance'], num_bins)
        self._extend_to_num_bins(self.depths, num_bins)
        self._extend_to_num_bins(self.matching['fuse_scores'], num_bins)
        
        return (
            output_indices, 
            self.matching['distance'], 
            self.matching['fuse_scores'], 
            self.depths
        )


class Associations(object):
    
    def __init__(self, distance=iou_distance, args=None):
        self.distance = distance
        self.strack_pool = None
        self.use_pseudo_depth = args.thresholds.pseudo_depth.add
        self.is_mot20 = args.mot20
        self.on_ground = args.uncertain.kalman.on_ground
        self.cmc = args.camera_motion.method
    
    def set_strack_pool(self, state):
        self.strack_pool = TrackUpdate.joint_stracks(state.tracks['tracked'], state.lost_stracks)
    
    def _indices(self, mapping, u_list):
        # convert indices from depth levels to original levels for tracks and dets
        return [*map(mapping.get, u_list)] if mapping and u_list else []
    
    def _assign(self, tracks, dets, match_thresh, is_fuse):
        self.distance.keywords['match_thresh'] = match_thresh
        self.distance.keywords['is_fuse'] = is_fuse
        # Compute distance between Kalman filter track prediction and current detections
        dists, det_scores = self.distance(tracks, dets)
        # Should low-confidence detections worsen the corresponding distances?
        if (not self.is_mot20) and is_fuse:
            dists = fuse_score(dists, dets, scores=det_scores)
        # matches, unmatched tracks, and unmatched detections
        matches, u_tracks, u_dets = linear_assignment(dists, thresh=match_thresh)
        return matches, u_tracks, u_dets
    
    def _update(self, state, tracks, dets, matches, frame_id):
        tracks_vel = []
        # update matched tracks
        for itracked, idet in matches:
            track, det = tracks[itracked], dets[idet]
            if track.state in (TrackState.Tracked, TrackState.Unconfirmed):
                track.update(det, frame_id)
                tracks_vel.append(track.mean - track.mean_old)
                state.tracks['activated'].append(track)
            else:
                track.re_activate(det, frame_id, new_id=False)
                state.tracks['refind'].append(track)
        return tracks_vel
    
    def apply(self, state, dets, match_thresh, frame_id, is_fuse=False, u_tracks=None, depths=None, level=None):
        if level == 0 and u_tracks != 'unconfirmed': 
            # Tracks are initial strack_pool
            tracks = u_tracks 
        elif isinstance(u_tracks, (tuple, np.ndarray, list)): 
            if self.cmc == 'kcm':
                strack_pool = u_tracks
            else:
                strack_pool = [t for t in u_tracks if t.state == TrackState.Tracked]
            tracks = strack_pool
            # Update strack_pool
            self.strack_pool = strack_pool
        else:
            # Tracks are temporally unconfirmed tracks
            tracks = state.tracks['unconfirmed']
            
        # If adding pseudo-depth info
        if self.use_pseudo_depth and depths is not None:
            tracks_ = TrackUpdate.deep_levels(tracks, depths)
            dets_ = TrackUpdate.deep_levels(dets, depths)
        else:
            tracks_ = [tracks]
            dets_ = [dets]
            
        # Collect residual tracks_ or dets_
        diff_len = len(tracks_) - len(dets_)
        if diff_len > 0:
            res_dets = []
            res_tracks = [t for lt in tracks_[-diff_len:] for t in lt]
        elif diff_len < 0:
            res_tracks = []
            res_dets = [d for ld in dets_[-abs(diff_len):] for d in ld]
        else:
            res_dets, res_tracks = [], []
            
        u_tracks_, u_dets_, tracks_vel = [], [], []    
        for ltracks, ldets in zip(tracks_, dets_):  
            ltracks += u_tracks_
            ldets += u_dets_
            # Assign detections to tracks
            matches, u_track_, u_det_ = self._assign(ltracks, ldets, match_thresh, is_fuse)
            # Update activated and rebirth tracks
            tracks_vel.extend(self._update(state, ltracks, ldets, matches, frame_id))
            u_tracks_ = [ltracks[i] for i in u_track_]
            u_dets_ = [ldets[i] for i in u_det_]
        
        u_tracks = u_tracks_ + res_tracks
        u_dets = u_dets_ + res_dets
        return u_tracks, u_dets, tracks_vel
    
    def unconfirm(self, state, u_detections, match_thresh, frame_id, u_levels=1, level=None):
        # Unmatched detections at score levels up to `levels` 
        u_detections = u_detections if u_levels == 'all' else u_detections[:u_levels]
        u_detections = [u_det for u_dets in u_detections for u_det in u_dets]
        u_unconfirmed, u_detections, *_ = self.apply(
            state, u_detections, match_thresh, frame_id, is_fuse=True, u_tracks='unconfirmed', level=level
        )
        # Unconfirmed tracks not matching any of the unmatched detections are removed
        state.destroy(u_unconfirmed, how='remove')
        return u_detections