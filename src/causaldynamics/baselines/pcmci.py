import fpcmci
import tigramite
from fpcmci.basics.utils import LabelType
from fpcmci.CPrinter import CPLevel
from fpcmci.FPCMCI import FPCMCI
from fpcmci.preprocessing.data import Data
from fpcmci.selection_methods.TE import TE, TEestimator
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI


class PCMCIPlus:
    """
    PCMCI+ baseline.

    Reference:
        [1] https://github.com/jakobrunge/tigramite/blob/master/tutorials/causal_discovery/tigramite_tutorial_pcmciplus.ipynb
    """

    def __init__(self, tau_max: int = 1, pc_alpha: float = 0.05):
        """Initialize regressor"""
        super(PCMCIPlus, self).__init__()
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha

    def run(self, X, verbosity: int = 0):
        """Estimate lagged adjacency graph"""
        self.estimator = PCMCI(
            dataframe=pp.DataFrame(X),
            cond_ind_test=ParCorr(significance="analytic"),
            verbosity=verbosity,
        )

        results = self.estimator.run_pcmciplus(
            tau_min=self.tau_max, tau_max=self.tau_max, pc_alpha=self.pc_alpha
        )
        self.adj_matrix = results["p_matrix"][..., -1] < self.pc_alpha


class FPCMCI:
    """
    FPCMCI baseline.

    Reference:
        [1] https://github.com/lcastri/fpcmci/blob/main/tutorials/02_FPCMCI.ipynb
    """

    def __init__(self, tau_max: int = 1, pc_alpha: float = 0.05):
        """Initialize regressor"""
        super(FPCMCI, self).__init__()
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha

    def run(self, X, verbosity: int = 0):
        """Estimate lagged adjacency graph"""
        self.estimator = fpcmci.FPCMCI.FPCMCI(
            Data(X),
            f_alpha=self.pc_alpha,
            pcmci_alpha=self.pc_alpha,
            min_lag=self.tau_max,
            max_lag=self.tau_max,
            sel_method=TE(TEestimator.Gaussian),
            val_condtest=ParCorr(significance="analytic"),
            verbosity=CPLevel.NONE,
        )

        _, results = self.estimator.run()
        self.adj_matrix = results.get_skeleton()[-1]
