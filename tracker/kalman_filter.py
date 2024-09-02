import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    The 8-dimensional state space
        x, y, w, h, vx, vy, vw, vh
    contains the bounding box center position (x, y), width w, height h,
    and their respective velocities.
    Object motion follows a constant velocity model. The bounding box location
    (x, y, w, h) is taken as direct observation of the state space (linear
    observation model).
    """

    def __init__(self, uncertain=False, is_on_ground=False, fps=1.0, is_nsa=False, **kwargs):
        
        if is_on_ground:
            ndim, dt = 2, 1/fps
            wx, wy = kwargs['wx'], kwargs['wy']
            self.vmax = kwargs['vmax']
            # Set noise matrix
            G = np.zeros((2 * ndim, ndim))
            G[0,0] = 0.5*dt*dt
            G[1,1] = 0.5*dt*dt
            G[2,0] = dt
            G[3,1] = dt
            Q0 = np.array([[wx, 0], [0, wy]])
            self._noise_mat = np.dot(np.dot(G, Q0), G.T)
            self.is_on_ground = True
        else:
            ndim, dt = 4, 1.0
            self._noise_mat = np.zeros((2 * ndim, 2 * ndim))
            self.is_on_ground = False
            
        self.uncertain = uncertain
        self.is_nsa = is_nsa
        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky ... until now that we "measure" uncertainty ;)
        self._std_weight_pos = (1. / 20) * np.ones((4,))
        self._std_weight_vel = (1. / 160) *np.ones((4,))

    def initiate(self, measurement, var_measurement=None):
        """Create track from unassociated measurement.
        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, w, h) with center position (x, y),
            width w, and height h.
        var_measurement : ndarray | None
            Variances of bounding box coordinates (x, y, w, h) with center
            position (x, y), width w, and height h.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        if self.is_on_ground:
            covariance = np.diag([1, 1, self.vmax**2/3.0, self.vmax**2/3.0])
            return mean, covariance
        
        whwh = np.tile(mean_pos[2:4], 2)
        std_pos = 2 * self._std_weight_pos * whwh
        std_vel = 10 * self._std_weight_vel * whwh     
        
        # Add var_measurement if not ignored
        if self.uncertain and var_measurement is not None:
            std_pos = np.c_[std_pos, np.sqrt(var_measurement)].min(axis=1)
        covariance = np.diag(np.r_[std_pos, std_vel]**2)
        
        return mean, covariance

    def project(self, mean, covariance, var_measurement=None, score=None, on_ground_cov=None):
        """Project state distribution to measurement space.
        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """
        whwh = np.tile(mean[2:4], 2)
        std = self._std_weight_pos * whwh
        innovation_cov = np.diag(np.square(std))
        
        if self.uncertain:
            if var_measurement is not None: # None within STrack.var_xywh
                innovation_cov = np.diag(var_measurement)
        
        if self.is_on_ground:
            innovation_cov = on_ground_cov
        
        if self.is_nsa and score is not None:
            innovation_cov *= 1 - score

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        covariance += innovation_cov
        return mean, 0.5 * (covariance + covariance.T) + 1e-7

    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        if self.is_on_ground:
            motion_cov = self._noise_mat
        else:
            whwh = np.tile(mean[:, 2:4], (1,2))
            std_pos = self._std_weight_pos[None,:] * whwh 
            std_vel = self._std_weight_vel[None,:] * whwh
            sqr = np.square(np.c_[std_pos, std_vel])
            motion_cov = np.asarray([np.diag(sqr[i]) for i in range(len(mean))])
        
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement, var_measurement=None, score=None):
        """Run Kalman filter correction step.
        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, w, h), where (x, y)
            is the center position, w the width, and h the height of the
            bounding box.
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        on_ground_cov = var_measurement if self.is_on_ground else None
        projected_mean, projected_cov = self.project(
            mean, covariance, var_measurement, score, on_ground_cov=on_ground_cov
        )
        try:
            chol_factor, lower = scipy.linalg.cho_factor(
                projected_cov, lower=True, check_finite=False)
            kalman_gain = scipy.linalg.cho_solve(
                (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
                check_finite=False).T
            innovation = measurement - projected_mean

            new_mean = mean + np.dot(innovation, kalman_gain.T)
            new_covariance = covariance - np.linalg.multi_dot((
                kalman_gain, projected_cov, kalman_gain.T))
            return new_mean, new_covariance
        except:
            return mean, covariance

    def gating_distance(self, mean, covariance, on_ground, measurements,
                        only_position=True, metric='maha', is_normalized=True):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance, on_ground_cov=on_ground.cov)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            if is_normalized:
                squared_maha += np.log(np.linalg.det(covariance))
            return squared_maha
        else:
            raise ValueError('invalid distance metric')