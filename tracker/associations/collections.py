from functools import partial
from .base import Associations
from ..matching import *


class BYTE(Associations):
    
    def __init__(self, distance=iou_distance, args=None):
        args.uncertain.kalman.measurements = False
        super().__init__(partial(distance), args=args)
        err_msg = 'scores.bins must be 2 in order to reproduce ByteTrack'
        assert args.thresholds.scores.bins == 2, err_msg
        
        
class USBYTE(Associations):
    
    def __init__(self, distance=score_iou_distance, args=None):
        args.uncertain.kalman.measurements = False
        args.uncertain.mask_distance = True
        super().__init__(partial(distance, args=args), args=args)


class UPBYTE(Associations):
    
    def __init__(self, distance=phase_iou_distance, args=None):
        args.uncertain.kalman.measurements = False
        super().__init__(partial(distance, args=args), args=args)


class UKBYTE(Associations):
    
    def __init__(self, distance=iou_distance, args=None):
        args.uncertain.kalman.measurements = True
        super().__init__(partial(distance), args=args)
                
        
class BoTSORT(Associations):
    
    def __init__(self, distance=iou_distance, args=None):
        args.uncertain.kalman.measurements = False
        args.camera_motion.method = 'fast-gmc'
        super().__init__(partial(distance), args=args)
        
        
class UKBoTSORT(Associations):
    
    def __init__(self, distance=iou_distance, args=None):
        args.uncertain.kalman.measurements = True
        args.camera_motion.method = 'fast-gmc'
        super().__init__(partial(distance), args=args)
        
        
class USBoTSORT(Associations):
    
    def __init__(self, distance=score_iou_distance, args=None):
        args.uncertain.kalman.measurements = False
        args.uncertain.mask_distance = True
        args.camera_motion.method = 'fast-gmc'
        super().__init__(partial(distance, args=args), args=args)


class UPBoTSORT(Associations):
    
    def __init__(self, distance=phase_iou_distance, args=None):
        args.uncertain.kalman.measurements = False
        args.camera_motion.method = 'fast-gmc'
        super().__init__(partial(distance, args=args), args=args)


class UKPBoTSORT(Associations):
    
    def __init__(self, distance=phase_iou_distance, args=None):
        args.uncertain.kalman.measurements = True
        args.camera_motion.method = 'fast-gmc'
        super().__init__(partial(distance, args=args), args=args)


class UBoTSORT(Associations):
    
    def __init__(self, distance=iou_distance, args=None):
        args.uncertain.kalman.measurements = False
        args.camera_motion.method = 'fast-ugmc'
        super().__init__(partial(distance), args=args)
        
        
class UKUBoTSORT(Associations):
    
    def __init__(self, distance=iou_distance, args=None):
        args.uncertain.kalman.measurements = True
        args.camera_motion.method = 'fast-ugmc'
        super().__init__(partial(distance), args=args)
        
        
class USUBoTSORT(Associations):
    
    def __init__(self, distance=score_iou_distance, args=None):
        args.uncertain.kalman.measurements = False
        args.uncertain.mask_distance = True
        args.camera_motion.method = 'fast-ugmc'
        super().__init__(partial(distance, args=args), args=args)


class UPUBoTSORT(Associations):
    
    def __init__(self, distance=phase_iou_distance, args=None):
        args.uncertain.kalman.measurements = False
        args.camera_motion.method = 'fast-ugmc'
        super().__init__(partial(distance, args=args), args=args)
                
        
class MABoTSORT(Associations):
    
    def __init__(self, distance=iou_distance, args=None):
        args.uncertain.kalman.measurements = False
        args.camera_motion.method = 'fast-gmc'
        args.thresholds.scores.bins = 2
        args.thresholds.matching.u_levels = 'all'
        super().__init__(partial(distance), args=args)
        
        
class USMABoTSORT(Associations):
    
    def __init__(self, distance=score_iou_distance, args=None):
        args.uncertain.kalman.measurements = False
        args.uncertain.mask_distance = True
        args.camera_motion.method = 'fast-gmc'
        args.thresholds.scores.bins = 2
        args.thresholds.matching.u_levels = 'all'
        super().__init__(partial(distance, args=args), args=args)


class UPMABoTSORT(Associations):
    
    def __init__(self, distance=phase_iou_distance, args=None):
        args.uncertain.kalman.measurements = False
        args.camera_motion.method = 'fast-gmc'
        args.thresholds.scores.bins = 2
        args.thresholds.matching.u_levels = 'all'
        super().__init__(partial(distance, args=args), args=args) 
                

class KCM(Associations):
    
    def __init__(self, distance=iou_distance, args=None):
        args.uncertain.kalman.measurements = False
        args.camera_motion.method = 'kcm'
        args.thresholds.matching.u_levels = 1
        super().__init__(partial(distance), args=args)


class SparseTrack(Associations):
    
    def __init__(self, distance=iou_distance, args=None):
        args.uncertain.kalman.measurements = False
        args.camera_motion.method = 'fast-gmc'
        args.thresholds.pseudo_depth.add = True
        args.thresholds.scores.bins = 2
        args.thresholds.pseudo_depth.depths = [1, 8]
        if args.thresholds.data == 'ablation_17':
            args.thresholds.matching.distance = [0.8, 0.4]
        elif args.thresholds.data == 'ablation_20':
            args.thresholds.matching.distance = [0.7, 0.3]
        super().__init__(partial(distance), args=args)


class USSparseTrack(Associations):
    
    def __init__(self, distance=score_iou_distance, args=None):
        args.uncertain.kalman.measurements = False
        args.camera_motion.method = 'fast-gmc'
        args.thresholds.pseudo_depth.add = True
        args.thresholds.scores.bins = 2
        args.thresholds.pseudo_depth.depths = [1, 8]
        args.thresholds.matching.distance = [0.8, 0.4]
        super().__init__(partial(distance, args=args), args=args)


class UPSparseTrack(Associations):
    
    def __init__(self, distance=phase_iou_distance, args=None):
        args.uncertain.kalman.measurements = False
        args.camera_motion.method = 'fast-gmc'
        args.thresholds.pseudo_depth.add = True
        args.thresholds.scores.bins = 2
        args.thresholds.pseudo_depth.depths = [1, 8]
        args.thresholds.matching.distance = [0.8, 0.4]
        super().__init__(partial(distance, args=args), args=args)
        

class MASparseTrack(Associations):
    
    def __init__(self, distance=iou_distance, args=None):
        args.uncertain.kalman.measurements = False
        args.camera_motion.method = 'fast-gmc'
        args.thresholds.pseudo_depth.add = True
        args.thresholds.scores.bins = 2
        args.thresholds.pseudo_depth.depths = [1, 8]
        if args.thresholds.data == 'ablation_17':
            args.thresholds.matching.distance = [0.8, 0.4]
        elif args.thresholds.data == 'ablation_20':
            args.thresholds.matching.distance = [0.7, 0.3]
        args.thresholds.matching.u_levels = 'all'
        super().__init__(partial(distance), args=args)


class UPMASparseTrack(Associations):
    
    def __init__(self, distance=phase_iou_distance, args=None):
        args.uncertain.kalman.measurements = False
        args.camera_motion.method = 'fast-gmc'
        args.thresholds.pseudo_depth.add = True
        args.thresholds.scores.bins = 2
        args.thresholds.pseudo_depth.depths = [1, 8]
        if args.thresholds.data == 'ablation_17':
            args.thresholds.matching.distance = [0.8, 0.4]
        elif args.thresholds.data == 'ablation_20':
            args.thresholds.matching.distance = [0.7, 0.3]
        args.thresholds.matching.u_levels = 'all'
        super().__init__(partial(distance, args=args), args=args)


class UCMC(Associations):
    
    def __init__(self, distance=mahalanobis_distance, args=None):
        args.uncertain.kalman.measurements = False
        args.uncertain.kalman.on_ground = True
        super().__init__(partial(distance), args=args)
    
    
""" Map to associations """
ASSOCIATIONS = { 
    'byte': BYTE,
    'us_byte': USBYTE,
    'up_byte': UPBYTE,
    'uk_byte': UKBYTE,
    'botsort': BoTSORT,
    'uk_botsort': UKBoTSORT,
    'us_botsort': USBoTSORT,
    'up_botsort': UPBoTSORT,
    'ukp_botsort': UKPBoTSORT,
    'ubotsort': UBoTSORT,
    'uk_ubotsort': UKUBoTSORT,
    'us_ubotsort': USUBoTSORT,
    'up_ubotsort': UPUBoTSORT,
    'mabotsort': MABoTSORT,
    'us_mabotsort': USMABoTSORT,
    'up_mabotsort': UPMABoTSORT,
    'kcm': KCM,
    'deep': SparseTrack,
    'us_deep': USSparseTrack,
    'up_deep': UPSparseTrack,
    'madeep': MASparseTrack,
    'up_madeep': UPMASparseTrack,
    'ucmc': UCMC,
}