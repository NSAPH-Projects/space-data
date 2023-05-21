from typing import Literal
import numpy as np
import networkx as nx
from scipy.linalg import cholesky, solve_triangular


def scale_variable(
    x: np.ndarray, scaling: Literal["unit", "standard"] = None
) -> np.ndarray:
    """Scales a variable according to the specified scale."""
    if scaling is None:
        return x

    match scaling:
        case "unit":
            return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
        case "standard":
            return (x - np.nanmean(x)) / np.nanstd(x)
        case _:
            raise ValueError(f"Unknown scale: {scaling}")


def transform_variable(
    x: np.ndarray, transform: Literal["log", "symlog", "logit"] = None
) -> np.ndarray:
    """Transforms a variable according to the specified transform."""
    if transform is None:
        return x

    match transform:
        case "log":
            return np.log(x)
        case "symlog":
            return np.sign(x) * np.log(np.abs(x))
        case "logit":
            return np.log(x / (1 - x))
        case _:
            raise ValueError(f"Unknown transform: {transform}")


def __find_best_gmrf_params(x: np.ndarray, graph: nx.Graph) -> np.ndarray:
    """Computes the BIC of a quadratic Gaussian model"""
    lams = 10 ** np.linspace(-3, 3, 20)
    e1 = np.array([e[0] for e in graph.edges])
    e2 = np.array([e[1] for e in graph.edges])
    L = nx.laplacian_matrix(graph).toarray()

    def solve(x, lam, L):
        Q = lam * L.copy()
        Q[np.diag_indices_from(Q)] += 1
        L = cholesky(Q, lower=True)
        z = solve_triangular(L, x, lower=True)
        beta = solve_triangular(L.T, z)
        return beta

    losses = []
    for lam in lams:
        # TODO: use sparse matrix/ugly dependencies
        beta = solve(x, lam, L)
        sig = np.std(x - beta)

        # compute loss assuming x ~ N(beta, sig**2)
        n = len(x)
        ll_loss = 0.5 * ((x - beta) / sig) ** 2 + n * np.log(2 * np.pi * (sig**2)) / 2

        # diffs ~ N(0, sig**2 / lam)
        diffs = beta[e1] - beta[e2]
        sigb = sig / np.sqrt(lam)
        m = len(diffs)
        prior_loss = 0.5 * (diffs / sigb) ** 2 + m * np.log(2 * np.pi * (sigb**2)) / 2

        losses.append(ll_loss + prior_loss)

    best_lam = lams[np.argmin(losses)]
    best_beta = solve(x, best_lam, L)
    best_sig = np.std(x - best_beta)

    return best_lam, best_sig


def generate_noise_like(residuals: np.ndarray, graph: nx.Graph, lam: float) -> np.ndarray:
    """Injects noise into residuals using a Gaussian Markov Random Field."""
    # find lambda value
    res_sig = np.nanstd(residuals)
    res_standard = residuals / res_sig
    best_lam, best_sig = __find_best_gmrf_params(res_standard, graph)

    # make spatial noise from GMRF
    Q = lam * nx.laplacian_matrix(graph).toarray()
    Z = np.random.normal(size=(Q.shape[0], len(residuals)))
    L = cholesky(Q, lower=True)
    spatial_noise = solve_triangular(L, Z, lower=True).T
    hetero_noise = np.random.normal(size=len(residuals)) * best_sig
    noise = (spatial_noise + hetero_noise) * res_sig

    # solve for beta
    return residuals + noise