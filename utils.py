import numpy as np
import pandas as pd
import os
from typing import Literal
import logging
import networkx as nx
from scipy.linalg import cholesky, solve_triangular
from pyDataverse.models import Datafile
from pyDataverse.api import NativeApi

LOGGER = logging.getLogger(__name__)

def upload_dataverse_data(
    data_path: str,
    data_description: str,
    dataverse_baseurl: str, 
    dataverse_pid: str,
    dataverse_token: str):
    """
    Upload data to the collection   
    Args:
        file_path (str): Filename 
        description (str): Data file description.
        token (str): Dataverse API Token.
    """
    api = NativeApi(dataverse_baseurl, dataverse_token)
    
    filename = os.path.basename(data_path)

    dataverse_datafile = Datafile()
    dataverse_datafile.set({
        "pid": dataverse_pid,
        "filename": filename,
        "description": data_description,
    })
    LOGGER.info("File basename: "+filename)
    
    resp = api.upload_datafile(
       dataverse_pid, data_path, dataverse_datafile.json())
    
    if resp.json()["status"] == "OK":
        LOGGER.info("Dataset uploaded.")
    else:
        LOGGER.error("Dataset not uploaded.")
        LOGGER.error(resp.json())


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
    """Select the best param using the penalized likelihood loss of a
    spatial GMLRF smoothing model."""
    lams = 10 ** np.linspace(-3, 1, 20)
    nodelist = np.array(graph.nodes)
    node2ix = {n: i for i, n in enumerate(nodelist)}
    e1 = np.array([node2ix[e[0]] for e in graph.edges])
    e2 = np.array([node2ix[e[1]] for e in graph.edges])
    L = nx.laplacian_matrix(graph).toarray()

    # solves the optiization problem argmin ||beta - x||^2 + lam * beta^T L beta
    def solve(x, lam, L):
        Q = lam * L.copy()
        Q[np.diag_indices_from(Q)] += 1
        L = cholesky(Q, lower=True)
        z = solve_triangular(L, x, lower=True)
        beta = solve_triangular(L.T, z)
        return beta

    losses = {}
    for lam in reversed(lams):
        # TODO: use sparse matrix/ugly dependencies
        beta = solve(x, lam, L)
        sig = np.std(x - beta)

        # compute loss assuming x ~ N(beta, sig**2)
        y_loss = 0.5 * ((x.values - beta) / sig) ** 2 + np.log(sig)

        # diffs ~ N(0, sig**2 / lam)
        l = (lam / sig**2)
        diff_loss = 0.5 * l * (beta[e1] - beta[e2]) ** 2 - 0.5 * np.log(l)

        penalty_loss = len(e1) * l + (1 / sig**2)

        # total_loss
        losses[lam] = y_loss.sum() + diff_loss.sum() + penalty_loss

    best_lam = min(lams, key=lambda l: losses[l])

    LOGGER.info(f"Best lambda: {best_lam:.4f}")
    losses_ = {np.round(k, 4): np.round(v, 4) for k, v in losses.items()}
    LOGGER.info(f"Losses: {losses_}")

    return best_lam


def generate_noise_like(x: pd.Series, graph: nx.Graph) -> np.ndarray:
    """Injects noise into residuals using a Gaussian Markov Random Field."""
    # find best smoothness param from penalized likelihood
    res_sig = np.nanstd(x)
    res_standard = x / res_sig
    res_graph = nx.subgraph(graph, x.index)
    best_lam = __find_best_gmrf_params(res_standard, res_graph)

    # make spatial noise from GMRF
    Q = best_lam * nx.laplacian_matrix(graph).toarray()
    Q[np.diag_indices_from(Q)] += 1
    Z = np.random.randn(Q.shape[0])
    L = cholesky(Q, lower=True)
    noise = solve_triangular(L, Z, lower=True).T
    noise = noise / noise.std() * res_sig

    return noise


def moran_I(x: pd.Series, A: np.ndarray) -> float:
    # Get the adjacency graph corresponding to the index of x
    x = x.values

    # input the mean of x when there nan values
    x[np.isnan(x)] = np.nanmean(x)
    
    # Compute mean of attribute values
    x_bar = np.mean(x)
    
    # Subtract mean from attribute values
    x_diff = x - x_bar

    # Compute denominator: sum of squared differences from mean
    denominator = np.sum(x_diff ** 2) + 1e-8

    # Compute numerator: sum of product of weight and pair differences from mean
    # Matrix multiplication to achieve vectorization
    numerator = np.sum(A * np.outer(x_diff, x_diff))

    # Compute Moran's I
    I = len(x) / np.sum(A) * (numerator / denominator)
    
    return float(I)