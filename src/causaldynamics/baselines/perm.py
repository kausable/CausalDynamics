from causallearn.search.PermutationBased.GRaSP import grasp as CL_GRASP

class GRASP:
    """
    Greedy relaxations of the sparsest permutation (GRaSP) algorithm.

    Reference:
        [1] Lam, W. Y., Andrews, B., & Ramsey, J. (2022, February). Greedy Relaxations of the Sparsest Permutation Algorithm. In The 38th Conference on Uncertainty in Artificial Intelligence.
    """

    def __init__(
        self, 
        tau_max: int = 1
    ):
        """Initialize regressor"""
        super(GRASP, self).__init__()
        self.tau_max = tau_max

    def run(self, X, verbosity: int = 0):
        """Estimate lagged adjacency graph"""
        G = CL_GRASP(X)
        self.adj_matrix = G.graph
        