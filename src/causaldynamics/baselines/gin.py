from causallearn.search.HiddenCausal.GIN.GIN import GIN as CL_GIN

class GIN:
    """
    Generalized Independence Noise (GIN) condition-based method.

    Reference:
        [1] Xie et al (2020, January). Generalized Independent Noise Condition for Estimating Latent Variable Causal Graphs. NeurIPS.
    """

    def __init__(
        self, 
        tau_max: int = 1
    ):
        """Initialize regressor"""
        super(GIN, self).__init__()
        self.tau_max = tau_max

    def run(self, X, verbosity: int = 0):
        """Estimate lagged adjacency graph"""
        G, K = CL_GIN(X)
        self.adj_matrix = G.graph

