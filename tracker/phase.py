import numpy as np
from collections import deque


class PhaseState(object):
    
    def __init__(self, w, var_w, conf, fps=30, doppler2phase=False, gm=0.99) -> None:
        # The state variable w is assumed to undergo SMH
        w, std_w = w, np.sqrt(var_w)
        self.w_mean = w
        self.w0 = w
        self.w = w
        self.doppler2phase = doppler2phase
        self.ws = deque([w])
        self.w_std = deque([std_w])
        self.conf = deque([conf])
        self.vshift = deque([0.0])
        self.is_freq_measured = False
        self.freq = 0
        self.T = float('inf')
        self.max_buffer_size = 60
        self.fps = fps
        self.t = 0
        self.t0 = 0
        self.gm = gm
    
    def update(
        self, 
        w_measured, 
        w_std_measured, 
        w_mean, 
        conf_measured, 
        vshift,
        reactivate
    ):
        if reactivate:
            PhaseState.__init__(self, w_measured, w_std_measured**2, conf_measured)
        
        self.t += 1 / self.fps
        self.ws.append(w_measured)
        self.w_std.append(w_std_measured)
        self.conf.append(conf_measured)
        
        vcurr, vold = vshift
        if abs(abs(vcurr - vold) - np.mean(self.vshift)) < 1.0:
            self.vshift.append(abs(vcurr - vold))
            cam_moved = False
        else:
            cam_moved = True
        
        went_back = abs(w_measured - self.w0) < np.asarray(self.w_std).mean() 
        if self.changes and went_back:
            # measure the avg frequency
            self.T = self.t - self.t0
            freq = (1 - self.gm) * self.freq + self.gm * (4 * np.pi / self.T)
            if self.doppler2phase:
                if cam_moved:
                    vcam = -(vcurr - vold)
                    freq = ((vcam - vcurr)/(vcam + vold)) * freq
            self.freq = freq
            # self.freq = freq
            self.is_freq_measured = True
            self.t0 = self.t
            self.w0 = w_measured
            
        if len(self.ws) > self.max_buffer_size:
            self.ws.popleft()
            self.w_std.popleft()
            self.conf.popleft()
            
        if len(self.vshift) > self.max_buffer_size:    
            self.vshift.popleft()
        
        self.w = w_measured    
        self.w_mean = w_mean
    
    @property
    def w_extremes(self):
        buffered_ws = np.asarray(self.ws)
        return [buffered_ws.min(), buffered_ws.max() + 1e-7]
    
    @property
    def changes(self):
        avg_w_std = np.asarray(self.w_std).mean() 
        w_min, w_max = self.w_extremes
        return avg_w_std > 0 and w_max - w_min > 3 * avg_w_std    
    
    @property
    def oscillating(self):
        # If current period T is close to implied by avg frequency
        if self.freq != 0:
            delta_T = abs(self.T - 2 * np.pi / self.freq) 
            return delta_T < 2 * (1 - self.gm)
        return False
    
    @property
    def cls_conf(self):
        return np.asarray(self.conf).mean()
    
    @property
    def angle(self):
        if self.is_freq_measured:
            return (self.freq * self.t) % 2*np.pi
        return np.random.uniform(0, 2*np.pi)
    
    @property
    def w_pred(self):
        if not self.is_freq_measured:
            return self.w_mean
        w_min, w_max = self.w_extremes
        d = np.sin(self.angle)
        return w_min + 0.5 * (w_max - w_min) * (1 + d)
    
    @staticmethod
    def current_det_angles(w_min_max, w):
        ''' Estimate angles of current detections by associating them to the stracks'''
        w_min, w_max = np.split(w_min_max, 2, axis=1)
        d = np.where(
            ~np.isclose(w_min, w_max), 
            -1 + 2 * (w[None,:] - w_min) / (w_max - w_min), 0.0 
        )
        return 2 * (np.pi/2 + np.arcsin(np.clip(d, -1, 1))) # [0,2pi]

    @staticmethod
    def compare_phases(stracks, dets):
        if len(stracks) == 0 or len(dets) == 0:
            empty = np.zeros((len(stracks), len(dets)), dtype=float)
            return 3 * [empty]
        
        # Create mask to compare only persons
        mask_both_person = np.outer(
            np.asarray([1 if st.cls == 0 else 0 for st in stracks]),
            np.asarray([1 if det.cls == 0 else 0  for det in dets])
        )
        # Estimate phases of current detections
        w_min_max = np.asarray([st.phase.w_extremes for st in stracks])
        w = np.asarray([det.phase.w for det in dets])
        dets_oscillating = np.asarray([det.phase.oscillating for det in dets])
        dets_current_angles = PhaseState.current_det_angles(w_min_max, w)
        # Estimate phases of tracked objects
        phase_vars = np.asarray(
            [[st.phase.angle, 1 * st.phase.oscillating] for st in stracks]
        )
        predicted_angles, tracks_oscillating = np.split(phase_vars, 2, axis=1)
        # Estimate statistic of oscillating boxes
        both_oscillating = np.outer(tracks_oscillating, dets_oscillating)
        p_oscillating = both_oscillating.sum() / both_oscillating.size # binomial mean
        
        p_both_person = both_oscillating * p_oscillating 
        p_both_person += (1 - both_oscillating) * (1 - p_oscillating) 
        # Bayes theorem
        p_oscillating_both_persons = both_oscillating * p_oscillating / p_both_person
        # Compute cosine similarity between tracked and new detection phases
        cosine_sim = 0.5 * (1 + np.cos(predicted_angles - dets_current_angles))
        return cosine_sim, mask_both_person, p_oscillating_both_persons