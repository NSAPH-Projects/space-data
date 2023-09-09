import logging
import os
import re
import shutil
from typing import Literal
from zipfile import ZipFile

import networkx as nx
import numpy as np
import pandas as pd
from omegaconf.listconfig import ListConfig
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


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
    x: np.ndarray,
    transform: list[str] | Literal["log", "symlog", "logit"] | None = None,
) -> np.ndarray:
    """Transforms a variable according to the specified transform."""
    if transform is None:
        return x
    elif isinstance(transform, (list, ListConfig)):
        # call recursively
        for t in transform:
            x = transform_variable(x, t)
        return x
    elif transform == "log":
        return np.log(x)
    elif transform == "symlog":
        return np.sign(x) * np.log(np.abs(x))
    elif transform == "logit":
        return np.log(x / (1 - x))
    elif transform.startswith("binary"):
        # regex to extract what's inside the parentheses, e.g., binary(10) -> 10
        cut_value = float(re.search(r"\((.*?)\)", transform).group(1))
        return np.where(x < cut_value, 0.0, 1.0)
    elif transform.startswith("gaussian_noise"):
        # regex to extract what's inside the parentheses, e.g., gaussian_noise(0.1) -> 0.1
        scaler = float(re.search(r"\((.*?)\)", transform).group(1))
        sig = np.nanstd(x)
        return x + np.random.normal(0, sig * scaler, x.shape)
    elif transform.startswith("qbinary"):
        value = float(re.search(r"\((.*?)\)", transform).group(1))
        quantile = np.quantile(x, value)
        return np.where(x < quantile, 0.0, 1.0)
    elif transform.startswith("affine"):
        args = re.search(r"\((.*?)\)", transform).group(1)
        b, m = [float(x) for x in args.split(",")]
        return b + m * x
    else:
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
        l = lam / sig**2
        diff_loss = 0.5 * l * (beta[e1] - beta[e2]) ** 2 - 0.5 * np.log(l)

        penalty_loss = len(e1) * l + (1 / sig**2)

        # total_loss
        losses[lam] = y_loss.sum() + diff_loss.sum() + penalty_loss

    best_lam = min(lams, key=lambda l: losses[l])

    logging.info(f"Best lambda: {best_lam:.4f}")
    losses_ = {np.round(k, 4): np.round(v, 4) for k, v in losses.items()}
    logging.info(f"Losses: {losses_}")

    return best_lam


def generate_noise_like_by_penalty(x: pd.Series, graph: nx.Graph) -> np.ndarray:
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


def generate_noise_like(
    x: np.ndarray, edge_list: np.ndarray, attempts: int = 10
) -> np.ndarray:
    """Injects noise into residuals using a Gaussian Markov Random Field."""
    n = len(x)
    nbrs_means = get_nbrs_means(x, edge_list)
    rho = get_nbrs_corr(x, edge_list, nbrs_means=nbrs_means)

    # 1. Build precision matrix
    # Arrays to hold the data, row indices, and column indices for Q
    data = []
    rows = []
    cols = []

    # Off-diagonal entries and compute degree for diagonal
    degree = np.zeros(n)
    for i, j in edge_list:
        data.extend([-rho, -rho])
        rows.extend([i, j])
        cols.extend([j, i])
        degree[i] += 1
        degree[j] += 1

    # Add diagonal entries
    data.extend(np.maximum(degree, 0.1))
    rows.extend(range(n))
    cols.extend(range(n))

    # build precision matrix
    Q = csc_matrix((data, (rows, cols)), shape=(n, n))
    factorization = splu(Q)

    best_result = np.inf
    best_corr = None
    best_attempt = None
    for _ in range(attempts):
        noise = factorization.solve(np.random.normal(size=n))
        noise_nbrs_means = get_nbrs_means(noise, edge_list)
        corr = np.corrcoef(noise, noise_nbrs_means)[0, 1]
        if np.abs(rho - corr) < best_result:
            best_result = np.abs(rho - corr)
            best_corr = corr
            best_attempt = noise

    # scale noise to have same variance as residuals
    noise = best_attempt / best_attempt.std() * np.nanstd(x)

    return noise


def get_nbrs_means(x: np.ndarray, edge_list: np.ndarray) -> np.ndarray:
    """Computes the mean of each node's neighbors."""
    nbrs = [[] for _ in range(x.shape[0])]
    for i, j in edge_list:
        nbrs[i].append(j)
        nbrs[j].append(i)

    xbar = np.nanmean(x)
    nbrs_means = np.zeros(len(x))
    for i in range(len(x)):
        if not nbrs[i]:
            nbrs_means[i] = xbar if np.isnan(x[i]) else x[i]
        else:
            valid = [x_j for x_j in x[nbrs[i]] if not np.isnan(x_j)]
            if valid:
                nbrs_means[i] = np.mean(valid)
            else:
                nbrs_means[i] = xbar if np.isnan(x[i]) else x[i]

    return nbrs_means


def get_nbrs_corr(
    x: np.ndarray, edge_list: np.ndarray, nbrs_means: np.ndarray | None = None
) -> float:
    """Computes the correlation between each node and its neighbors."""
    if nbrs_means is None:
        nbrs_means = get_nbrs_means(x, edge_list)
    x_ = x.copy()
    x_[np.isnan(x_)] = nbrs_means[np.isnan(x_)]
    rho = np.corrcoef(x_, nbrs_means)[0, 1]
    return float(rho)


def moran_I(x: np.ndarray, edge_list: np.ndarray) -> float:
    x = x.copy()

    xbar = np.nanmean(x)
    nbrs_means = get_nbrs_means(x, edge_list)

    x_ = x.copy()
    x_[np.isnan(x_)] = nbrs_means[np.isnan(x_)]

    # Subtract mean from attribute values
    x_diff = x_ - xbar

    # Compute numerator: sum of product of weight and pair differences from mean
    src_diff = x_diff[edge_list[:, 0]]
    dst_diff = x_diff[edge_list[:, 1]]
    numerator = np.sum(src_diff * dst_diff) * len(x_diff)

    # Compute denominator: sum of squared differences from mean
    denominator = np.sum(x_diff**2) * len(edge_list)

    return float(numerator / denominator)


def double_zip_folder(folder_path, output_path):
    # Create a temporary zip file
    shutil.make_archive(output_path, "zip", folder_path)

    # Zip the temporary zip file
    zipzip_path = output_path + ".zip.zip"
    with ZipFile(zipzip_path, "w") as f:
        f.write(output_path + ".zip")

    # Remove the temporary zip file
    os.remove(output_path + ".zip")

    return zipzip_path


def sort_dict(d: dict) -> dict[str, float]:
    return {
        str(k): float(v) for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True)
    }


def spatial_train_test_split(
    graph: nx.Graph, init_frac: float, levels: int, buffer: int
):
    logging.info(f"Selecting tunning split removing {levels} nbrs from val. pts.")

    # make dict of neighbors from graph
    node_list = np.array(graph.nodes())
    n = len(node_list)
    nbrs = {node: set(graph.neighbors(node)) for node in node_list}

    # first find the centroid of the tuning subgraph
    num_tuning_centroids = int(init_frac * n)
    tuning_nodes = np.random.choice(n, size=num_tuning_centroids, replace=False)
    tuning_nodes = set(node_list[tuning_nodes])

    # not remove all neighbors of the tuning centroids from the training data
    for _ in range(levels):
        tmp = tuning_nodes.copy()
        for node in tmp:
            for nbr in nbrs[node]:
                tuning_nodes.add(nbr)
    tuning_nodes = list(tuning_nodes)

    # buffer
    buffer_nodes = set(tuning_nodes.copy())
    for _ in range(buffer):
        tmp = buffer_nodes.copy()
        for node in tmp:
            for nbr in nbrs[node]:
                buffer_nodes.add(nbr)
    buffer_nodes = list(set(buffer_nodes))

    return tuning_nodes, buffer_nodes


def unpack_covariates(groups: dict) -> list[str]:
    covariates = []
    for c in groups:
        if isinstance(c, dict) and len(c) == 1:
            covariates.extend(next(iter(c.values())))
        elif isinstance(c, str):
            covariates.append(c)
        else:
            msg = "covar group must me dict with a single element or str"
            logging.error(msg)
            raise ValueError(msg)

    return covariates
