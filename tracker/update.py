from pathlib import Path
from collections import namedtuple

import numpy as np
from omegaconf import OmegaConf

from .kalman_filter import KalmanFilter
from .basetrack import BaseTrack, TrackState
from .phase import PhaseState
from .camera_motion import CameraMotionCompensation
from .associations.base import TrackUpdate, Thresholds
from .associations.collections import ASSOCIATIONS
from .config.utils.mapper import Mapper


class STrack(BaseTrack):
    
    shared_kalman = KalmanFilter()
    
    def __init__(
        self, 
        tlwh, 
        score, 
        cls, 
        var_xywhs, 
        fps=30, 
        uncertain=False, 
        seq=None, 
        dataset=None,
        doppler2phase=None
    ):
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.mean_old, self.covariance_old = None, None
        self.phase = PhaseState(
            self._tlwh[2], var_xywhs[2], score, fps, doppler2phase=doppler2phase
        )
        self.is_activated = False

        self.score = score
        self.var_score = var_xywhs[-1]
        self.cls = cls
        self._var_xywh = var_xywhs[:-1]
        self.tracklet_len = 0
        
        config_path = Path(__file__).parent / 'config/utils'
        if dataset.lower() != 'kitti': # FIXME: find calib data for kitti
            cam_para_file = config_path / f'cam_para/{dataset}/{seq}.txt'
            # Original mapping is modified to include "measurement" of R^uv
            self.ground_mapper = Mapper(cam_para_file, dataset)
            z, R = self.ground_mapper.mapto(self._tlwh, self._var_xywh, uncertain)
        else:
            z, R = np.zeros((1, 2)), np.zeros((2, 2))
        self.on_ground = namedtuple('on_ground', 'pos cov')(z.squeeze(), R)
        self.vshift = namedtuple('vshift', 'vcurr vold')
    
    @classmethod
    def set_is_on_ground(cls, args):
        is_on_ground = args.uncertain.kalman.on_ground
        kwargs = OmegaConf.to_container(args.ucmc)
        cls.shared_kalman = KalmanFilter(is_on_ground=is_on_ground, **kwargs)
    
    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        
        if not self.kalman_filter.is_on_ground:
            new_track_obs = self.tlwh_to_xywh(self._tlwh)
            new_track_err = self._var_xywh
        else:
            new_track_obs = self.on_ground.pos
            new_track_err = self.on_ground.cov
        
        self.mean, self.covariance = self.kalman_filter.initiate(
            new_track_obs, new_track_err
        )
        # Store track state for KMC
        self.mean_old = self.mean.copy()
        self.covariance_old = self.covariance.copy()
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        if not self.kalman_filter.is_on_ground:
            new_track_obs = self.tlwh_to_xywh(new_track.tlwh)
            new_track_err = new_track.var_xywh
        else:
            new_track_obs = new_track.on_ground.pos
            new_track_err = new_track.on_ground.cov
        
        # Store old track state for KMC
        self.mean_old = self.mean.copy()
        self.covariance_old = self.covariance.copy()
            
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track_obs, new_track_err, new_track.score
        )
        self.phase.update(
            new_track._tlwh[2], 
            new_track._var_xywh[2], 
            self.mean[2], 
            new_track.score, 
            vshift=self.vshift(self.mean[4], self.mean_old[4]),
            reactivate=True
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        # Transfer box details
        self._tlwh = new_track._tlwh
        self._var_xywh = new_track._var_xywh

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        # Store old track state for KMC
        self.mean_old = self.mean.copy()
        self.covariance_old = self.covariance.copy()

        if not self.kalman_filter.is_on_ground:
            new_track_obs = self.tlwh_to_xywh(new_track.tlwh)
            new_track_err = new_track.var_xywh
        else:
            new_track_obs = new_track.on_ground.pos
            new_track_err = new_track.on_ground.cov
            
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track_obs, new_track_err, new_track.score
        )
        self.phase.update(
            new_track._tlwh[2], 
            new_track._var_xywh[2], 
            self.mean[2], 
            new_track.score, 
            vshift=self.vshift(self.mean[4], self.mean_old[4]),
            reactivate=False
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        # Transfer box details
        self._tlwh = new_track._tlwh
        self._var_xywh = new_track._var_xywh
    
    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][-1] = 0
                    multi_mean[i][-2] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
    
    @staticmethod
    def multi_kcm(tracks_vel, u_tracks):
        if tracks_vel and u_tracks:
            # Get camera velocity in state space
            tracks_vel = np.asarray(tracks_vel).mean(0)
            match_tracks_acc = tracks_vel[4:]
            vel = np.r_[4*[0], match_tracks_acc]
            # Restore track states before prediction, correcting with camera velocity
            multi_mean = np.array([st.mean_old.copy() - vel for st in u_tracks]) # Galileo
            multi_covariance = np.array([st.covariance_old.copy() for st in u_tracks]) #FIXME: try current cov
            # Predict again, with the corresponding correction and put tracks back into pool
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                u_tracks[i].mean = mean
                u_tracks[i].covariance = cov
    
    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)# np.kron
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_ugmc(stracks, H=np.eye(3, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            H1 = H[:2, :2]
            h2 =  H[:2, 2]
            h3T = H[2, :2]
            h4 = H[2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                xy, wh, vxy, vwh = np.split(mean, 4)
                # Corrected xy and vxy
                num_transf_xy = H1.dot(xy) + h2
                den_transf_xy = h3T.dot(xy) + h4
                Gxy = (den_transf_xy * H1 - np.outer(num_transf_xy, h3T)) / (den_transf_xy**2)
                cxy = num_transf_xy / den_transf_xy
                cvxy = Gxy.dot(vxy)
                
                # Corrected wh and vwh
                xyl = xy - 0.5* wh
                xyr = xy + 0.5* wh
                num_transf_xyl = H1.dot(xyl) + h2
                den_transf_xyl = h3T.dot(xyl) + h4
                num_transf_xyr = H1.dot(xyr) + h2
                den_transf_xyr = h3T.dot(xyr) + h4
                cxyl = num_transf_xyl / den_transf_xyl
                cxyr = num_transf_xyr / den_transf_xyr
                Gxyl = (den_transf_xyl * H1 - np.outer(num_transf_xyl, h3T)) / (den_transf_xyl**2)
                Gxyr = (den_transf_xyr * H1 - np.outer(num_transf_xyr, h3T)) / (den_transf_xyr**2)
                cvxyl = Gxyl.dot(vxy - 0.5 * vwh)
                cvxyr = Gxyr.dot(vxy + 0.5 * vwh)
                cwh = cxyr - cxyl
                cvwh = cvxyr - cvxyl
                
                # Corrected mean and cov
                mean = np.r_[cxy, cwh, cvxy, cvwh]
                G8x8 = np.kron(np.eye(4, dtype=float), Gxy)
                cov = G8x8.dot(cov).dot(G8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov
                
    @staticmethod
    def multi_ucmc(stracks, affine=np.eye(2, 3)):
        if len(stracks) > 0:
            
            M = affine[:,:2]
            T = np.zeros((2,1))
            T[0,0] = affine[0,2]
            T[1,0] = affine[1,2]
            
            for track in stracks:
                u, v = track.ground_mapper.xy2uv(*track.mean[:2])
                w, h = track._tlwh[2:]

                p_center = np.array([[u], [v-h/2]])
                p_wh = np.array([[w], [h]])
                p_center = np.dot(M, p_center) + T
                p_wh = np.dot(M, p_wh)

                u = p_center[0,0]
                v = p_center[1,0]+p_wh[1,0]/2
                xy, _ = track.ground_mapper.uv2xy(np.array([[u],[v]]), np.eye(2))
                track.mean[:2] = xy.squeeze()

    @property
    def is_on_ground(self):
        return self.kalman_filter.is_on_ground
    
    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def from_ground_tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        u, v = self.ground_mapper.xy2uv(*self.mean[:2])
        w, h = self._tlwh[2:] # box size from latest det
        return np.asarray([u - 0.5*w, v-h, w, h])
    
    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret
    
    @property
    def var_xywh(self):
        """Estimates of xywh variances.
        """
        return self._var_xywh.copy()
    
    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret
    
    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        if ret.ndim == 2:
            ret[:,2:] -= ret[:,:2]
        else:
            ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret
    
    @property
    def deep_vec(self):
        """Convert bounding box to format `((top left, bottom right)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        cx = ret[0] + 0.5 * ret[2]
        y2 = ret[1] + ret[3]
        length = 2000 - y2
        assert length > 0
        return np.asarray([cx, y2, length], dtype=float)

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class Tracker(object):
    
    def __init__(self, args):
        BaseTrack.clear_count()
        
        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(args.frame_rate / 30.0 * args.track_buffer)
        
        # Select associations and thresholds to apply
        self.associations = ASSOCIATIONS[args.association](args=args)
        cmc_method = args.camera_motion.method
        self.camera_motion = CameraMotionCompensation(STrack, method=cmc_method)
        self.thresh = Thresholds(cmc_method, STrack, **args.thresholds)
        
        # Select Kalman filter
        STrack.set_is_on_ground(args)
        self.kalman_filter = KalmanFilter(
            is_on_ground=args.uncertain.kalman.on_ground,
            uncertain=args.uncertain.kalman.measurements,
            is_nsa=args.uncertain.kalman.nsa,
            fps=args.frame_rate,
            **OmegaConf.to_container(args.ucmc)
        )
        # Define the state of the tracks (temporal and persistent)
        self.state = TrackUpdate()

    def update(self, output_results, dataset=None, seq_name=None):
         
        self.frame_id += 1
        if self.frame_id == 1:
            self.pre_img = None
        
        ''' Reset temporal state of tracks '''
        self.state.reset()
        
        ''' Add previously detected and unactivated tracklets to temporal state'''
        self.state.confirm()

        ''' Define strack pool: tracks currently activated '''
        self.associations.set_strack_pool(self.state)
        
        ''' Predict the new track state from Kalman filter '''
        STrack.multi_predict(self.associations.strack_pool)
        
        ''' Read current detections from object detector, removing low-confident ones '''
        bboxes, scores, classes, var_bboxes, img = TrackUpdate.read_detections(
            output_results, self.thresh.low_score
        )
        
        ''' Define and perform binning procedure for the cascade association '''
        bins_score = self.thresh.define_binning_strategy(var_bboxes[:,-1]) 
        association_levels = zip(*self.thresh.bin_detections(scores, bins_score))
        
        u_tracks = self.associations.strack_pool
        u_detections = []
        tracks_vel = []
        
        ''' Associate predictions to detections at multiple levels '''
        for level, (highest, match_thresh, is_fuse, depths) in enumerate(association_levels):
            # Read current high-confidence detections
            dets = bboxes[highest]
                        
            # Apply camera motion compensation
            to_all_levels = (
                self.args.camera_motion.method == 'kcm' or
                self.args.camera_motion.all_levels or
                level == 0 # all or first
            )
            if self.args.camera_motion.method is not None and to_all_levels: 
                self.camera_motion.compensate(
                    self.state, self.associations, img, self.pre_img, 
                    dets, tracks_vel, u_tracks
                )
                self.pre_img = img

            # Convert detection bounding boxes to STrack format
            detections = TrackUpdate.detection2strack(STrack, dets,
                *(scores[highest], classes[highest], var_bboxes[highest]),
                fps=self.args.frame_rate, dataset=dataset, seq=seq_name,
                uncertain=self.kalman_filter.uncertain, 
                doppler2phase=self.args.camera_motion.doppler2phase
            )
            # Match detections with tracks
            u_tracks, u_detection, tracks_vel = self.associations.apply(
                self.state, detections, match_thresh, self.frame_id, 
                is_fuse=is_fuse, u_tracks=u_tracks, depths=depths, level=level
            )
            # Collect unmatched detections at different score levels (bins)
            u_detections.append(u_detection)
            
        # Tracks left unmatched are marked lost
        self.state.destroy(u_tracks, how='loose')
        
        # Unconfirmed tracks with no matches are marked removed
        u_detections = self.associations.unconfirm(
            self.state, u_detections, self.thresh.confirm, self.frame_id, 
            u_levels=self.args.thresholds.matching.u_levels, level=level
        )
        ''' Init new stracks '''
        self.state.create(u_detections, self.kalman_filter, self.thresh.new_track, self.frame_id)
        
        ''' Update persistent state '''
        output_stracks = self.state.update(self.frame_id, self.max_time_lost)

        return output_stracks