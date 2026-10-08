import warnings

import lingam
import numpy as np

warnings.filterwarnings("ignore")


class VARLiNGAM:
    """
    VARLiNGAM baseline.

    Reference:
        [1] https://github.com/cdt15/lingam/blob/master/examples/VARLiNGAM.ipynb
    """

    def __init__(self, tau_max: int = 1):
        """Initialize regressor"""
        super(VARLiNGAM, self).__init__()
        self.tau_max = tau_max

    def run(self, X, verbosity: int = 0):
        """Estimate lagged adjacency graph"""
        self.estimator = lingam.VARLiNGAM(lags=self.tau_max)

        self.estimator.fit(X)
        self.adj_matrix = self.estimator.adjacency_matrices_[self.tau_max]
        self.adj_matrix = np.abs(self.adj_matrix) > 0
