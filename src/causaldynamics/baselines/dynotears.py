import numpy as np
import pandas as pd
from causalnex.structure import dynotears


class DYNOTEARS:
    """
    DYNOTEARS baseline.

    Reference:
        [1] https://github.com/mckinsey/causalnex/blob/develop/causalnex/structure/dynotears.py
    """

    def __init__(
        self, 
        tau_max: int = 1,
        lambda_w: float = 0.2,
        lambda_a: float = 0.1,
        
    ):
        """Initialize regressor"""
        super(DYNOTEARS, self).__init__()
        self.tau_max = tau_max
        self.lambda_w = lambda_w
        self.lambda_a = lambda_a

    def run(self, X, verbosity: int = 0):
        """Estimate lagged adjacency graph"""
        self.estimator = dynotears.from_pandas_dynamic(
            pd.DataFrame(X), 
            self.tau_max,
            lambda_w=self.lambda_w,
            lambda_a=self.lambda_a
        )
        self.adj_matrix = self._postprocess()

    def _postprocess(self):
        """Retrieve adjacency matrix given fitted model"""
        adj_view = self.estimator.adj

        source_keys = [
            key for key in adj_view.keys() if key.endswith(f"_lag{self.tau_max}")
        ]
        target_keys = [key for key in adj_view.keys() if key.endswith("_lag0")]

        source_keys = sorted(source_keys, key=lambda x: int(x.split("_")[0]))
        target_keys = sorted(target_keys, key=lambda x: int(x.split("_")[0]))

        n = len(target_keys)
        adj_matrix = np.zeros((n, n))

        for s_key in source_keys:
            # Extract the source variable index from the key (e.g., "2_lag1" --> 2)
            i = int(s_key.split("_")[0])
            for t_key, attributes in adj_view[s_key].items():
                # Only consider targets at lag0
                if t_key.endswith("_lag0"):
                    j = int(t_key.split("_")[0])
                    adj_matrix[i, j] = attributes["weight"]

        return np.abs(adj_matrix) > 0
