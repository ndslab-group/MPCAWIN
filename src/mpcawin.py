from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skmultiflow.drift_detection import KSWIN
import pandas as pd
import numpy as np
import math
import queue
import random

class MPCAWIN():
    _DETECT_STRATEGIES = ['alert_rate','detector']

    def __init__(self,
                 min_window_size=10,
                 max_window_size=10,
                 coef_std=2,
                 detect_strategy='detector',
                 alert_window_size=20,
                 alpha=0.5,
                 detector=KSWIN(alpha=0.009, window_size=20,stat_size=5),
                 random_seed=20
                 ):
        """
        An unsupervised multi-dimensional concept drift detection method (MPCAWIN) based on the statistical distribution of input features from 'HOIDS: Concept Drift Aware Hybrid Online Intrusion Detection System'.

        Parameters
        ----------
        min_window_size: int (default=10)
            Minimum size of sliding window to extract the statistical information of the measure of dissimilarity.

        max_window_size: int (default=10)
            Max size of sliding window to extract the statistical information of the measure of dissimilarity.

        coef_std: int (default=2)
            The statistical information extracted from the sliding window is the threshold used 
            to decide whether a batch is too dissimilar from the statistical distribution. The threshold is calculated as 
            np.mean(sliding_window)+coef_std*np.std(sliding_window).

        detect_strategy: str (default='detector')
            This parameter is used to determine how to analyze the one-dimensional distribution, it is recommended to use a valid one-dimensional detector.

        alert_window_size: int (default=20)
            Size of alert sliding window.

        alpha: float (default=0.5)
            This parameter is used if the detect strategy is 'alert_rate'.
            The percentage of times inside the alert sliding window that triggers drift detection if the threshold is exceeded.
            It is equivalent to the alpha parameter of many drift detectors and adjusts the sensitivity of the detector.
            A value near 0.0 makes the detector more sensitive to noise but potentially faster at detecting drift.
            A value near 1.0 makes the detector more robust to noise but delays drift detection.

        detector: BaseDriftDetector (default=None)
            It is used if the detect strategy is 'detector'.
            A BaseDriftDetection of skmultiflow.drift_detection.base_drift_detector.

        random_seed: int (default=20)
            A seed for the random operations.
            Set this value for reproducibility of experiments.

        Notes
        -----
        More detail in Sec. 4.1 and Fig.6 of 'HOIDS: Concept Drift Aware Hybrid Online Intrusion Detection System' paper.

        References
        ----------
        """
        if min_window_size == 0:
            raise AttributeError("Invalid min window size value")
        if max_window_size < min_window_size:
            raise AttributeError("Invalid max window size. max window size value must be greater than or equal to the min window size value")
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.coef_std = coef_std
        if detect_strategy not in self._DETECT_STRATEGIES:
            raise AttributeError("Invalid detect_strategy: {}\n"
                                 "Valid options: {}".format(detect_strategy,
                                                            self._DETECT_STRATEGIES))
        self.detect_strategy = detect_strategy
        if self.detect_strategy == 'alert_rate':
            if alpha > 0 and alpha <= 1:
                self.alpha = alpha
            else:
                raise AttributeError("Invalid alpha value. Valid range is (0.0;1.0])")
            if alert_window_size == 0:
                raise AttributeError("Invalid alert window size value")
        else:
            self.detector = detector
            self.random_seed = random_seed
            if self.random_seed != 'None':
                random.seed(self.random_seed)
                np.random.seed(self.random_seed)

        #self.change_detected = False
        self.window = queue.Queue(self.max_window_size)
        self.alert_window_size = alert_window_size
        self.alert_window = queue.Queue(self.alert_window_size)
        self.threshold = None
        #self.iter = 0
        self.actual_transform_concept = None
        self.actual_concept_eigenvalues = None
        self.last_angle = None
        self.last_noise_angle = None
        
    def reset(self):
        """ Resets the change detector parameters.
        """
        #self.change_detected = False
        self.window = queue.Queue(self.max_window_size)
        if self.detect_strategy == 'detector':
            self.detector.reset()
        else:
            self.alert_window = queue.Queue(self.max_window_size*2)
        self.threshold = None
        #self.iter = 0
        self.actual_transform_concept = None
        self.actual_concept_eigenvalues = None
        #self.last_angle = None
           
    def set_concept(self, input_value):
        """ Set the input batch of data as a actual concpet

        Transforms a input batch of data with Standard Scaler and PCA trained on it.
        Extracts the vector of eigenvalues of the batch.
        Resets the change detector parameters.

        Parameters
        ----------
        input_value: {array-like, DataFrame} of shape (n_samples, n_features)
            Internally, it will be converted to DataFrame=pd.DataFrame.
            New batch of data the detector should compare with last concept.
        """
        self.reset()
        concept = pd.DataFrame(input_value)
        self.scaler = StandardScaler(copy=True)
        self.pca = PCA(copy=True)
        self.scaler.fit(X=concept)
        scaler_concept = self.scaler.transform(X=concept)
        self.pca.fit(X=scaler_concept)
        self.actual_transform_concept = self.pca.transform(X=scaler_concept)
        self.actual_concept_eigenvalues = self.pca.explained_variance_

    def add_element_and_detected_change(self, input_value):
        """ Add a new batch of data and get detected change

        Transforms a new batch of data with Standard Scaler and PCA trained on the last batch of data set as the current concept.
        Extracts the vector of eigenvalues of the new batch and calculates the angle in degrees between 
        the vector of eigenvalues of the new batch and that of the last concept.
        Applies Statistical Hypothesis Test:
         (1) calculates the threshold from the sliding window values;
         (2) check whether or not the new angle exceeds the threshold;
         (3) add 0 if the new angle does not exceed the threshold, add 1 if the new angle exceeds the threshold 
         to the alert sliding window if the detect_strategy is 'alert_rate' otherwise to the detector.
         (4) detect drift based on the alpha value if if the detect_strategy is 'alert_rate' otherwise based on the detector test.
        Add the new angle to the sliding window if does not exceed the threshold.

        Parameters
        ----------
        input_value: {array-like, DataFrame} of shape (n_samples, n_features)
            Internally, it will be converted to DataFrame=pd.DataFrame.
            New batch of data the detector should compare with last concept.

        Returns
        -------
        bool
            Whether or not a drift occurred

        """
        self.last_noise_angle = None
        if self.actual_concept_eigenvalues is not None:
            concept = pd.DataFrame(input_value)
            scaler_new_concept = self.scaler.transform(concept)
            self.pca.fit(scaler_new_concept)
            v2 = self.pca.explained_variance_
            ps = np.inner(self.actual_concept_eigenvalues, v2)
            v1_norm = np.linalg.norm(self.actual_concept_eigenvalues)
            v2_norm = np.linalg.norm(v2)
            a = ps / (v1_norm * v2_norm)
            angle = math.degrees(math.acos(a))
            self.last_angle = round(angle,2)
            if self.threshold is not None:
                result = int(self.last_angle>self.threshold)
                if self.detect_strategy == 'alert_rate':
                    if self.alert_window.full():
                        self.alert_window.get()
                        self.alert_window.put(result)
                        if list(self.alert_window.queue).count(1)>=math.ceil(self.alert_window.qsize()*self.alpha):
                            self.set_concept(input_value)
                            return True
                    else:
                        self.alert_window.put(result)
                else:
                    self.detector.add_element(result)
                    if self.detector.detected_change():
                        self.set_concept(input_value)
                        return True
                if result==0:
                    if self.window.full():
                        self.window.get()
                        self.window.put(self.last_angle)
                    else:
                        self.window.put(self.last_angle)
                    self.threshold = round(np.mean(self.window.queue)+self.coef_std*np.std(self.window.queue),2)
                else:
                    self.last_noise_angle = self.last_angle
            else:
                self.window.put(self.last_angle)
                if self.window.qsize()==self.min_window_size:
                    self.threshold = round(np.mean(self.window.queue)+self.coef_std*np.std(self.window.queue),2)
        else:
            self.set_concept(input_value)
        return False
