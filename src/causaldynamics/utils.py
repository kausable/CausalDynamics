import random
from datetime import datetime

import numpy as np
import torch


def set_rng_seed(seed=None, verbose=False):
    """
    Set the random seed for reproducibility across numpy, torch, and random.

    Parameters
    ----------
    seed : int, optional
        The random seed to use. If None, no seed will be set.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.cuda.manual_seed_all(seed + 4)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if verbose:
            print(f"Random seed set to {seed} for reproducibility")


def get_timestamp():
    """
    Get the current timestamp in the format YYYYMMDD_HHMMSS.

    Returns
    -------
    str : The current timestamp in the format YYYYMMDD_HHMMSS.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def allocate_elements_based_on_ratios(n, ratios):
    """Allocate a total number of elements into groups based on fractional proportions.

    Parameters
    ----------
    n : int
        Total number of elements to allocate. Must be positive.
    ratios : list of float
        List of ratios specifying the proportional allocation for each group.
        Values must be positive.

    Returns
    -------
    list of int
        List containing the number of elements allocated to each group.
        The sum of all elements equals n.

    Raises
    ------
    ValueError
        If n is not positive, frac is empty, contains invalid values,
        or does not sum to approximately 1.

    Notes
    -----
    The function handles rounding errors by randomly distributing any remaining
    elements to maintain the total count n.
    """
    # Validate inputs
    if n < 0:
        raise ValueError(f"n must be a positive integer or 0. Got: n={n}")
    if not ratios:
        raise ValueError(f"Ratio list 'ratios' cannot be empty. Got: ratios={ratios}")
    if any(r < 0 for r in ratios):
        raise ValueError(
            f"All elements in 'ratios' must be positive. Got: ratios={ratios}"
        )

    # Calculate fractions from ratios
    sum_ratios = sum(ratios)
    frac = [r / sum_ratios for r in ratios]

    # Allocate elements based on fractions
    data = [int(f * n) for f in frac]
    n_mod = n - sum(data)
    # Adjust for rounding errors (if any)
    max_attempts = 1000  # Prevent infinite loops

    for _ in range(max_attempts):
        if n_mod <= 0:
            break

        idx = torch.randint(0, len(data), (1,)).item()
        data[idx] += 1
        n_mod -= 1
    else:
        import warnings

        warnings.warn(
            f"Could not fully distribute {n_mod} remaining elements after {max_attempts} attempts"
        )

    return data


def check_confounders(ds):
    """
    Check a dataset if it contains confounders.

    Parameters
    ----------
    ds : xarray.Dataset
        Preliminary xarray dataset representing one SCM graph.

    Returns
    -------
    is_confounded : bool
        Flag to check if the system has confounding variable or not.
    """
    # Detect primary dim ("node" or "dim")
    for primary in ("node", "dim"):
        if primary in ds.dims:
            break
    else:
        raise ValueError("Couldn't find 'node' or 'dim' dimension in ds")

    in_dim = f"{primary}_in"
    out_dim = f"{primary}_out"

    # Identify confounding sets (w/ children >= 2 on the off-diagonal; first variable)
    coord_in = ds.adjacency_matrix.coords[in_dim]
    coord_out = ds.adjacency_matrix.coords[out_dim]
    off_diag = coord_in != coord_out

    child_counts = ((ds.adjacency_matrix != 0) & off_diag).sum(dim=in_dim)

    conf_indices = np.where(child_counts.values >= 2)[0]
    is_confounded = False if conf_indices.size == 0 else True
    return is_confounded


def process_confounders(ds):
    """
    Postprocess a dataset by simulating an unobserved confounders.
    This induces psuedo-A and dropped timeseries of the identified confounding variable.

    Parameters
    ----------
    ds : xarray.Dataset
        Preliminary xarray dataset representing one SCM graph.

    Returns
    -------
    xarray.Dataset : The processed dataset simulating unobserved confounder cases.
    """
    # Detect primary dim ("node" or "dim")
    for primary in ("node", "dim"):
        if primary in ds.dims:
            break
    else:
        raise ValueError("Couldn't find 'node' or 'dim' dimension in ds")

    in_dim = f"{primary}_in"
    out_dim = f"{primary}_out"

    # Identify confounding sets (w/ children >= 2 on the off-diagonal; first variable)
    coord_in = ds.adjacency_matrix.coords[in_dim]
    coord_out = ds.adjacency_matrix.coords[out_dim]
    off_diag = coord_in != coord_out

    child_counts = ((ds.adjacency_matrix != 0) & off_diag).sum(dim=in_dim)

    conf_indices = np.where(child_counts.values >= 2)[0]
    if conf_indices.size == 0:
        ## No confounder found --> return original
        return ds

    conf_indices = int(conf_indices[0])

    # Drop confounding variables
    ds_confounded = ds.drop_sel(
        {in_dim: conf_indices, out_dim: conf_indices, primary: conf_indices}
    )

    return ds_confounded
