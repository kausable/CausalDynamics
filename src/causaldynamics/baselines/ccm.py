import typing
import warnings

import numpy as np
import scipy
from jaxtyping import Float
from numpy import ndarray
from statsmodels.tsa import stattools
from tqdm import tqdm


class TSCI:
    """
    TSCI baseline.

    Reference:
        [1] https://github.com/KurtButler/tangentspace
    """

    def __init__(self, tau_max: int = 1, corr_thres: float = 0.8):
        """Initialize regressor"""
        super(TSCI, self).__init__()
        self.tau_max = tau_max
        self.corr_thres = corr_thres

    def run(self, X, verbosity: int = 0):
        """Estimate lagged adjacency graph"""
        T, D = X.shape
        X_source = X[: -self.tau_max, :]
        X_target = X[self.tau_max :, :]
        self.adj_matrix = np.random.randn(D, D)

        # Loop over all pairs (target, source)
        for source in range(D):
            for target in range(D):
                self.adj_matrix[source, target] = self._estimate_pair(
                    X_source[:, source], X_target[:, target]
                )

        self.adj_matrix = self.adj_matrix > self.corr_thres

    def _estimate_pair(self, x, y):
        """Compute pairwise correlation"""
        x_signal = x.reshape(-1, 1)
        y_signal = y.reshape(-1, 1)

        # Get embedding hyperparameters and create delay embeddings
        tau_x = self.tau_max
        tau_y = self.tau_max
        Q_x = 2  # X embedding dim
        Q_y = 2  # Y embedding dim

        x_state = delay_embed(x_signal, tau_x, Q_x)
        y_state = delay_embed(y_signal, tau_y, Q_y)
        truncated_length = min(
            x_state.shape[0], y_state.shape[0]
        )  # Omit earliest samples
        x_state = x_state[-truncated_length:]
        y_state = y_state[-truncated_length:]

        # Get velocities with (centered) finite differences
        dx_dt = discrete_velocity(x_signal)
        dy_dt = discrete_velocity(y_signal)

        # Delay embed velocity vectors
        dx_state = delay_embed(dx_dt, tau_x, Q_x)
        dy_state = delay_embed(dy_dt, tau_y, Q_y)
        dx_state = dx_state[-truncated_length:]
        dx_state = dx_state
        dy_state = dy_state[-truncated_length:]
        dy_state = dy_state

        ############################
        ####    Perform TSCI    ####
        ############################
        r_x2y, _ = tsci_nn(
            x_state,
            y_state,
            dx_state,
            dy_state,
            fraction_train=0.8,
            use_mutual_info=False,
        )

        return np.mean(r_x2y)


def next_pow_two(n: int) -> int:
    """Gets the next power of two greater than `n`. Code from [1].

    [1] https://dfm.io/posts/autocorr/

    Args:
        n (int): number to get next power of two of

    Returns:
        int: next power of two greater than `n`
    """
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(
    x: Float[ndarray, " N"], norm: bool = True
) -> Float[ndarray, " N"]:
    """Computes the autocorrelation function (ACF) of a signal. Code from [1].

    [1] https://dfm.io/posts/autocorr/

    Args:
        x (Float[ndarray, &quot; N&quot;]): signal values
        norm (bool, optional): whether to normalize to autocorrelation. Defaults to True.

    Raises:
        ValueError: if `len(x.shape) > 1`

    Returns:
        Float[ndarray, &quot; N&quot;]: the ACF of `x`
    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf


def auto_window(taus: Float[ndarray, " N"], c: float) -> int:
    """Automated windowing procedure following Sokal (1989). Code from [1].

    [1] https://dfm.io/posts/autocorr/

    Args:
        taus (Float[ndarray, &quot; N&quot;]): autocorrelation times
        c (float): constant used in method

    Returns:
        int: window length
    """
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return int(np.argmin(m))
    return len(taus) - 1


def autocorr_new(y: Float[ndarray, " N"], c=5.0) -> float:
    """Returns the autocorrelation time

    [1] https://dfm.io/posts/autocorr/

    Args:
        y (Float[ndarray, &quot; N&quot;]): _description_
        c (float, optional): _description_. Defaults to 5.0.

    Returns:
        float: _description_
    """
    y = np.atleast_2d(y)
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def false_nearest_neighbors(
    y: Float[ndarray, " T"],
    tau: int,
    fnn_tol: float = 0.01,
    Q_max: int = 20,
    rho: float = 17.0,
) -> int:
    """Computes a heuristic embedding dimension of `y` with time lag `tau` using the false nearest neighbors (FNN) algorithm.

    Args:
        y (Float[ndarray, &quot; T&quot;]): time series to compute FNN for.
        tau (int): time lag to use, for example as computed by `lag_select`.
        fnn_tol (float, optional): tolerance for the amount of false nearest neighbors. Defaults to 0.01.
        Q_max (int, optional): maximum allowed embedding dimension. Defaults to 20.
        rho (float, optional): magic number as proposed in the FNN paper. Defaults to 17.0.

    Returns:
        int: embedding dimension
    """
    Q = 1
    fnn_flag = False

    # Q is repeatedly increased until the number of false nearest neighbors falls below `fnn_tol`
    while not fnn_flag:
        Q += 1
        if Q > Q_max:
            warnings.warn("FNN did not converge.")
            return Q_max

        M1 = delay_embed(y, tau, Q)
        M2 = delay_embed(y, tau, Q + 1)

        M1 = M1[: M2.shape[0]]
        fnn = np.zeros(M1.shape[0])

        kdtree = scipy.spatial.KDTree(M1)

        for n in range(M1.shape[0]):
            _, ids = kdtree.query(M1[n, :], 2)
            # We may consider only the nearest neighbors, whose index is `ids[1]`
            Rd = np.linalg.norm(M1[ids[1], :] - M1[n, :], 2) / np.sqrt(Q)
            # Nearest neighbors will be much closer in the lower dimension
            # so ||M_2[n] - M_2[NN]||_2 / ||M_1[n] - M_1[NN]||_2 will be large
            fnn[n] = np.linalg.norm(M2[n, :] - M2[ids[1], :], 2) > rho * Rd

        if np.mean(fnn) < fnn_tol:
            fnn_flag = True

    return Q


def discrete_velocity(x: Float[ndarray, "T 1"], smooth=False) -> Float[ndarray, "T 1"]:
    """Gets the discrete derivative of a time series.
    This simply wraps `np.gradient`, so it uses a second order finite difference method
    everywhere except the boundaries.

    Args:
        x (Float[ndarray, &quot; T&quot;]): time series array

    Returns:
        Float[ndarray, &quot; T&quot;]: gradient
    """
    if smooth:
        return scipy.signal.savgol_filter(x, 5, 2, deriv=1, axis=0)
    else:
        return np.gradient(x, axis=0)


def lag_select(x: Float[ndarray, " T"], theta: float = 0.5, max_tau: int = 100) -> int:
    """Selects a time lag based on the autocorrelation function (ACF).

    Args:
        x (Float[ndarray, &quot; T&quot;]): the time series.
        theta (float, optional): the desired autocorrelation to fall under. Defaults to 0.5.
        max_tau (int, optional): maximum allowable time lag. Defaults to 100.

    Returns:
        int: selected time lag
    """
    # Calculate ACF, first on a default sub=timeseries
    acf = stattools.acf(x - x.mean())
    # Calculate for the entire time series if all values are about the threshold
    if np.all(acf >= theta):
        acf = stattools.acf(x, min(max_tau, len(x) - 1))

    tau = int(np.argmax(acf < theta))
    if tau == 0:
        tau = max_tau
    return tau


def delay_embed(
    x: Float[ndarray, "T 1"], lag: int, embed_dim: int
) -> Float[ndarray, "T embed_dim"]:
    """Computes the delay embedding with lag `tau` and dimension `embed_dim` of `x`

    Args:
        x (Float[ndarray, &quot; T&quot;]): time series
        lag (int): lag for delay embedding
        embed_dim (int): desired dimension

    Returns:
        Float[ndarray, &quot; T&quot;, &quot; embed_dim&quot;]: delay embedding of `x` with lag `tau` and embedding dimension `embed_dim`
    """
    num_x = x.shape[0] - (embed_dim - 1) * lag
    embed_list = []

    for i in range(embed_dim):
        embed_list.append(
            x[
                (embed_dim - 1) * lag
                - (i * lag) : (embed_dim - 1) * lag
                - (i * lag)
                + num_x
            ].reshape(-1, x.shape[1])
        )
    return np.concatenate(embed_list, axis=-1)


def tsci_nn(
    x_state: Float[ndarray, "T Q_x"],  # noqa: F722
    y_state: Float[ndarray, "T Q_y"],  # noqa: F722
    dx_state: Float[ndarray, "T Q_x"],  # noqa: F722
    dy_state: Float[ndarray, "T Q_y"],  # noqa: F722
    fraction_train: float = 0.8,
    lib_length: int = -1,
    use_mutual_info=False,
) -> typing.Tuple[Float[ndarray, "T 1"], Float[ndarray, "T 1"]]:  # noqa: F722
    """Performs Tangent Space Causal Inference (TSCI) with the nearest neighbors approach.

    Args:
        x_state (Float[ndarray, &quot;T Q_x&quot;]): delay embedding of signal $
        y_state (Float[ndarray, &quot;T Q_y&quot;]): delay embedding of signal $y$
        dx_state (Float[ndarray, &quot;T Q_x&quot;]): vector field of `x_state`
        dy_state (Float[ndarray, &quot;T Q_y&quot;]): vector field of `y_state`
        fraction_train (float, optional): fraction of training data in train/test split. Defaults to 0.8.
        lib_length (int, optional): library length to test with. If negative, defaults to `fraction_train * len(x_state.shape[0])`. Defaults to -1.

    Returns:
        typing.Tuple[Float[ndarray, &quot;T 1&quot;], Float[ndarray, &quot;T 1&quot;]]: correlation coefficients for causal directions $X \\to Y$ and $Y \\to X$
    """
    Q_x = dx_state.shape[1]
    Q_y = dy_state.shape[1]
    N_samples = dx_state.shape[0]
    N_train = int(fraction_train * N_samples)

    if lib_length < 0:
        lib_length = N_train

    # the pushforward dx vectors should look like the dy vectors
    x_pushforward = np.zeros_like(dy_state[N_train:])

    # initialize a KDTree to do repeated nearest-neighbor lookup
    K = 3 * Q_x
    kdtree = scipy.spatial.KDTree(x_state[:lib_length])

    ########################
    ## Pushforward X -> Y ##
    ########################
    # For each point in the test set, we find the Jacobian and pushfoward the corresponding `dx_state` sample
    for n in tqdm(range(N_train, x_state.shape[0]), leave=False):
        # Query points and get displacement vectors
        _, ids = kdtree.query(x_state[n, :], K)
        x_tangents = x_state[ids, :] - x_state[n, :]
        y_tangents = y_state[ids, :] - y_state[n, :]

        # The Jacobian is the least-squares solution mapping x displacements to y displacements
        lstsq_results = scipy.linalg.lstsq(x_tangents, y_tangents)
        J = lstsq_results[0]

        # Pushforward is a vector-Jacobian product
        x_pushforward[n - N_train, :] = dx_state[n, :] @ J

    ########################
    ## Pushforward Y -> X ##
    ########################
    # For each point in the test set, we find the Jacobian and pushfoward the corresponding `dy_state` sample
    y_pushforward = np.zeros_like(dx_state[N_train:])
    K = 3 * Q_y

    kdtree = scipy.spatial.KDTree(y_state[:lib_length])
    for n in tqdm(range(N_train, y_state.shape[0]), leave=False):
        # Query points and get displacement vectors
        _, ids = kdtree.query(y_state[n, :], K)
        x_tangents = x_state[ids, :] - x_state[n, :]
        y_tangents = y_state[ids, :] - y_state[n, :]

        # The Jacobian is the least-squares solution mapping y displacements to x displacements
        lstsq_results = scipy.linalg.lstsq(y_tangents, x_tangents)
        J = lstsq_results[0]

        # Pushforward is a vector-Jacobian product
        y_pushforward[n - N_train, :] = dy_state[n, :] @ J

    ########################
    # Compute correlations #
    ########################
    if use_mutual_info:
        score_x2y = KSGEnsembleFirstEstimator(neighborhoods=(10,)).estimate(
            dx_state[N_train:], y_pushforward
        )

        score_y2x = KSGEnsembleFirstEstimator(neighborhoods=(10,)).estimate(
            dy_state[N_train:], x_pushforward
        )
    else:
        dotprods = np.sum(dx_state[N_train:] * y_pushforward, axis=1)
        mags1 = np.sum(dx_state[N_train:] * dx_state[N_train:], axis=1)
        mags2 = np.sum(y_pushforward * y_pushforward, axis=1)
        score_x2y = dotprods / np.sqrt(mags1 * mags2 + 1e-16)

        dotprods = np.sum(dy_state[N_train:] * x_pushforward, axis=1)
        mags1 = np.sum(dy_state[N_train:] * dy_state[N_train:], axis=1)
        mags2 = np.sum(x_pushforward * x_pushforward, axis=1)
        score_y2x = dotprods / np.sqrt(mags1 * mags2 + 1e-16)

    return score_x2y, score_y2x
