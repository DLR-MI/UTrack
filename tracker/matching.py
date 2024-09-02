import numpy as np
import lap
from fuzzy_cython_bbox import bbox_overlaps as bbox_ious
from fuzzy_cython_bbox import fuzzy_bbox_ious, disambiguate_ious
from .phase import PhaseState


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1]))
        )
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(tracks_tlbrs, detections_tlbrs):
    ious = np.zeros(
        (len(tracks_tlbrs), len(detections_tlbrs)),
        dtype=float
    )
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(tracks_tlbrs, dtype=float),
        np.ascontiguousarray(detections_tlbrs, dtype=float)
    )
    return ious


def fuzzy_ious(
    tracks_tlbrs,
    detections_tlbrs,
    tracks_xywh_stds,
    detections_xywh_stds
):
    ious = np.zeros(
        (len(tracks_tlbrs), len(detections_tlbrs)),
        dtype=float
    )
    if ious.size == 0:
        return ious, ious

    ious, ious_std = fuzzy_bbox_ious(
        np.ascontiguousarray(tracks_tlbrs, dtype=float),
        np.ascontiguousarray(detections_tlbrs, dtype=float),
        np.ascontiguousarray(tracks_xywh_stds, dtype=float),
        np.ascontiguousarray(detections_xywh_stds, dtype=float)
    )
    return ious, ious_std


def iou_distance(tracks, detections, match_thresh=None, is_fuse=None):
    tracks_tlbrs = [track.tlbr for track in tracks]
    detections_tlbrs = [track.tlbr for track in detections]
        
    _ious = ious(tracks_tlbrs, detections_tlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix, None


def fuzzy_iou_distance(
    tracks,
    detections,
    disambiguator,
    thresh_iou,
    iou_std,
    thresh_std=0.01
):
    tracks_tlbrs = [track.tlbr for track in tracks]
    detections_tlbrs = [track.tlbr for track in detections]
    tracks_xywh_std = [np.sqrt(track.var_xywh) for track in tracks]
    detections_xywh_std = [np.sqrt(track.var_xywh) for track in detections]
    det_scores = np.asarray([track.score for track in detections])
        
    # Compute ious with their respective errors    
    _ious, _ious_std = fuzzy_ious(
        tracks_tlbrs,
        detections_tlbrs,
        tracks_xywh_std,
        detections_xywh_std
    )
    if 0 in _ious.shape:
        return _ious, _ious
    
    # Use computed iou errors or provided hyperparameter
    ious_std = iou_std if isinstance(iou_std, float) else _ious_std
    det_scores = det_scores[None,:].repeat(_ious.shape[0], axis=0).astype(np.float64)
    
    # Disambiguate ious by rearranging rows where ious are unresolved according
    # to the order imposed by the sorted reference    
    reference, mask = disambiguator(tracks, detections, ious=_ious)
    
    f_ious, f_scores, was_ambiguous = disambiguate_ious(
        _ious, ious_std, reference, det_scores,
        thresh_iou, thresh_std, perm_scores=False
    )
    # disambiguate_msg = '++++++++++ Disambiguated ious ++++++++++' if was_ambiguous else ''
    # if disambiguate_msg: print(disambiguate_msg)
        
    cost_matrix = 1 - _ious
    f_cost_matrix = 1 - f_ious
    f_dist = np.where(mask, f_cost_matrix, cost_matrix)
    f_scores = np.where(mask, f_scores, det_scores)

    return f_dist, f_scores


def phase_iou_distance(tracks, detections, args=None, match_thresh=None, is_fuse=None):

    def disambiguator(tracks, detections, ious=None):
        cosine_sim, mask_both_person, conf_w_oscillates =\
            PhaseState.compare_phases(tracks, detections)
        phase_sim = conf_w_oscillates * cosine_sim
        return phase_sim, mask_both_person
    
    ious_std = (
        args.uncertain.std_iou if isinstance(args.uncertain.std_iou, float) else 'auto'
    )
    return fuzzy_iou_distance(tracks, detections, disambiguator, match_thresh, ious_std)


def score_iou_distance(tracks, detections, args=None, match_thresh=None, is_fuse=None):

    def disambiguator(tracks, detections, ious=None):
        mask = np.ones_like(ious) * args.uncertain.mask_distance
        track_sizes = np.asarray([track.tlwh for track in tracks], dtype=float)
        track_sizes = np.prod(track_sizes[:, 2:], axis=1)
        track_sizes = (track_sizes - track_sizes.mean()) / (track_sizes.std() + 1e-7)
        det_sizes = np.asarray([det.tlwh for det in detections], dtype=float)
        det_sizes = np.prod(det_sizes[:, 2:], axis=1)
        det_sizes = (det_sizes - det_sizes.mean()) / (det_sizes.std() + 1e-7)
        T, D = np.meshgrid(track_sizes, det_sizes)
        return np.exp(-0.5 * (np.square(T - D))).T, mask
    
    ious_std = args.uncertain.std_iou if isinstance(args.uncertain.std_iou, float) else 'auto'
    return fuzzy_iou_distance(tracks, detections, disambiguator, match_thresh, ious_std)


def mahalanobis_distance(tracks, detections, match_thresh=None, is_fuse=None):
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    if 0 in cost_matrix.shape:
        return cost_matrix
    
    measurements = np.asarray([det.on_ground.pos for det in detections])
    for row, track in enumerate(tracks):
        maha_distance = track.kalman_filter.gating_distance(
            track.mean, track.covariance, track.on_ground, measurements)
        cost_matrix[row,:] = maha_distance
    return cost_matrix, None


def fuse_score(cost_matrix, detections, scores=None):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    if scores is not None:
        det_scores = scores
    else:
        det_scores = np.array([det.score for det in detections])
        det_scores = det_scores[None,:].repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost