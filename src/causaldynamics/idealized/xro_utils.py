import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


###########################################
def time_axis(tim, offset=False, freq="MS"):
    if freq == "MS":
        xtime = tim.dt.year + (tim.dt.month - 0.5) / 12.0
        if offset:
            xtime = xtime - tim.dt.year[0]
    elif freq == "D":
        xtime = tim.dt.year + (tim.dt.dayofyear - 0.5) / 365
        if offset:
            xtime = xtime - tim.dt.year[0]
    return xtime


def plot_above_below_shading(
    x_ds,
    above=0.5,
    below=-0.5,
    c="black",
    above_c="#E5301D",
    below_c="#301D80",
    alpha_fill=0.8,
    xtime=None,
    ax=None,
    **kwargs,
):
    if xtime is None:
        x_axis = time_axis(x_ds.time)
    else:
        x_axis = xtime

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax.plot(x_axis, x_ds, c=c, **kwargs)
    ax.fill_between(
        x_axis,
        x_ds,
        above,
        where=x_ds >= above,
        interpolate=True,
        color=above_c,
        alpha=alpha_fill,
    )
    ax.fill_between(
        x_axis,
        x_ds,
        below,
        where=x_ds <= below,
        interpolate=True,
        color=below_c,
        alpha=alpha_fill,
    )
    return ax


def plot_fill_between(
    x_ds,
    dim="member",
    c="orangered",
    alpha=0.2,
    xtime=None,
    ax=None,
    option=None,
    **kwargs,
):

    if xtime is None:
        x_axis = time_axis(x_ds.time)
    else:
        x_axis = xtime

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    if option is None:
        x_lo = x_ds.mean(dim) - x_ds.std(dim)
        x_up = x_ds.mean(dim) + x_ds.std(dim)
    else:
        x_lo = x_ds.quantile(option, dim=dim)
        x_up = x_ds.quantile(1 - option, dim=dim)

    ax.plot(x_axis, x_ds.mean(dim), c=c, **kwargs)
    ax.fill_between(x_axis, x_lo, x_up, fc=c, alpha=alpha)
    return ax


def compute_skew(da, dim):
    mean_da = da.mean(dim=dim)
    std_da = da.std(dim=dim)
    skewness = ((da - mean_da) / std_da) ** 3
    return skewness.mean(dim=dim)


@xr.register_dataarray_accessor("skew")
@xr.register_dataset_accessor("skew")
class SkewAccessor:
    """
    calculate the skewness using xarray
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, dim="time"):
        """
        Compute the skewness along the given dimension.
        """
        if isinstance(self._obj, xr.DataArray):
            return compute_skew(self._obj, dim)

        elif isinstance(self._obj, xr.Dataset):
            skews = {}
            for name, da in self._obj.data_vars.items():
                skews[name] = compute_skew(da, dim)
            return xr.Dataset(skews)


######################################################################


def pmtm(ds_x, dim="time", dt=1 / 12.0, nw=4, cl=0.95, nfft=None, scale_by_freq=True):
    """
    Thomson’s multitaper power spectral density (PSD) estimate,
    PSD significance level estimate, Confidence Intervals

    PSD estimate modified from Peter Huybers's matlab code, pmtmPH.m
    PSD significance level estimate modified from NCAR/NCL specx_ci

    Parameters
    ----------
    x : numpy array
        Time series to analyze
    dt : float
        Sampling interval, default is 1.
    nw : float
        The time bandwidth product, available
    cl : float
        Confidence interval to calculate and display
    nfft: int
        Default

    Returns
    -------
    x_P : xarray DataArray
        PSD estimate via the multi-taper method.
    x_P_sig : xarray DataArray
        PSD significance level estimate.
    x_ci : xarray DataArray
        Confidence intervals.
    """
    P_ds, s_ds, Psig_ds, ci_ds = xr.apply_ufunc(
        _pmtm,
        ds_x,
        input_core_dims=[[dim]],
        output_core_dims=[
            ["freq"],
            ["freq"],
            ["freq"],
            [
                "freq",
                "bound",
            ],
        ],
        kwargs={
            "dt": dt,
            "nw": nw,
            "cl": cl,
            "nfft": nfft,
            "scale_by_freq": scale_by_freq,
        },
        vectorize=True,
    )
    if len(s_ds.dims) == 1:
        P_ds = P_ds.assign_coords({"freq": s_ds})
        Psig_ds = Psig_ds.assign_coords({"freq": s_ds})
        ci_ds = ci_ds.assign_coords({"freq": s_ds})
    else:
        mem_dim = list(s_ds.dims)[0]
        P_ds = P_ds.assign_coords({"freq": s_ds.isel({mem_dim: 0}).values})
        Psig_ds = Psig_ds.assign_coords({"freq": s_ds.isel({mem_dim: 0}).values})
        ci_ds = ci_ds.assign_coords({"freq": s_ds.isel({mem_dim: 0}).values})
    return P_ds, Psig_ds, ci_ds


def _pmtm(x, dt=1, nw=4, cl=0.95, nfft=None, scale_by_freq=True):
    """
    numpy of pmtm
    """

    from scipy.signal import windows
    from scipy.stats import chi2

    if nfft is None:
        nfft = np.shape(x)[0]

    nx = np.shape(x)[0]
    k = min(np.round(2.0 * nw), nx)
    k = int(max(k - 1, 1))
    s = np.arange(0, 1 / dt, 1 / (nfft * dt))

    # Compute the discrete prolate spheroidal sequences
    [E, V] = windows.dpss(nx, nw, k, return_ratios=True)
    E = E.T

    # Compute the windowed DFTs.
    Pk = np.abs(np.fft.fft(E * x[:, np.newaxis], nfft, axis=0)) ** 2

    # Iteration to determine adaptive weights
    if k > 1:
        sig2 = np.dot(x[np.newaxis, :], x[:, np.newaxis])[0][0] / nx  # power
        # initial spectrum estimate
        P = ((Pk[:, 0] + Pk[:, 1]) / 2)[:, np.newaxis]
        Ptemp = np.zeros((nfft, 1))
        P1 = np.zeros((nfft, 1))
        tol = 0.0005 * sig2 / nfft
        a = sig2 * (1 - V)

        while np.sum(np.abs(P - P1) / nfft) > tol:
            b = np.repeat(P, k, axis=-1) / (
                P * V[np.newaxis, :] + np.ones((nfft, 1)) * a[np.newaxis, :]
            )
            wk = (b**2) * (np.ones((nfft, 1)) * V[np.newaxis, :])
            P1 = (np.sum(wk * Pk, axis=-1) / np.sum(wk, axis=-1))[:, np.newaxis]

            Ptemp = np.empty_like(P1)
            Ptemp[:] = P1
            P1 = np.empty_like(P)
            P1[:] = P
            P = np.empty_like(Ptemp)
            P[:] = Ptemp

        # Determine equivalent degrees of freedom, see Percival and Walden 1993.
        v = (
            2 * np.sum((b**2) * (np.ones((nfft, 1)) * V[np.newaxis, :]), axis=-1) ** 2
        ) / np.sum((b**4) * (np.ones((nfft, 1)) * V[np.newaxis, :] ** 2), axis=-1)

    else:
        P = np.empty_like(Pk)
        P[:] = Pk
        v = 2 * np.ones((nfft, 1))

    select = (np.arange(0, (nfft + 1) / 2.0)).astype("int")
    P = P[select].flatten()
    s = s[select].flatten()
    v = v[select].flatten()

    # Whether the resulting density values should be scaled by the scaling frequency, which gives density in units of 1/day, 1/year, or 1/Hz.
    # This allows for integration over the returned frequency values
    if scale_by_freq:
        P *= dt

    # Chi-squared 95% confidence interval
    # approximation from Chamber's et al 1983; see Percival and Walden p.256, 1993
    ci = np.empty((np.shape(v)[0], 2))
    ci[:, 0] = 1.0 / (1 - 2 / (9 * v) - 1.96 * np.sqrt(2 / (9 * v))) ** 3
    ci[:, 1] = 1.0 / (1 - 2 / (9 * v) + 1.96 * np.sqrt(2 / (9 * v))) ** 3

    # red noise significance levels
    # PSD significance level estimate modified from NCAR/NCL specx_ci
    alpha = np.corrcoef(x[:-1], x[1:])[0, 1]
    dof = v
    P_red = (1 - alpha**2) / (1 + alpha**2 - 2 * alpha * np.cos(2 * np.pi * s * dt))
    # rescale
    scale = np.sum(P) / np.sum(P_red)
    quantile = chi2.ppf(cl, dof)
    P_sig = P_red * scale * quantile / dof

    x_s = xr.DataArray(s, dims=["freq"], coords={"freq": s})
    x_P = xr.DataArray(P, dims=["freq"], coords={"freq": s})
    x_P_sig = xr.DataArray(P_sig, dims=["freq"], coords={"freq": s})
    x_ci = xr.DataArray(ci, dims=["freq", "bound"], coords={"freq": s, "bound": [0, 1]})
    return x_P, x_s, x_P_sig, x_ci


def _unique_ordered(elements):
    """Return a list of unique elements in the order they first appeared."""
    seen = set()
    result = []
    for element in elements:
        if element not in seen:
            seen.add(element)
            result.append(element)
    return result


def legend_combo(ax, reverse=False, **kwargs):
    """
    create legend combo the same name together
    """
    handler, labeler = ax.get_legend_handles_labels()
    hd = []
    labli = _unique_ordered(labeler)
    for lab in labli:
        comb = [h for h, l in zip(handler, labeler) if l == lab]
        hd.append(tuple(comb))
    if reverse:
        ax.legend(hd[::-1], labli[::-1], **kwargs)
    else:
        ax.legend(hd, labli, **kwargs)
