import re
import numpy as np
import pandas as pd
import os
import shutil
from zipfile import ZipFile
from typing import Literal
from omegaconf.listconfig import ListConfig
import logging
import networkx as nx
from scipy.linalg import cholesky, solve_triangular
from pyDataverse.models import Datafile
from pyDataverse.api import NativeApi, DataAccessApi

LOGGER = logging.getLogger(__name__)


def download_dataverse_data(
    filename: str,
    dataverse_baseurl: str,
    dataverse_pid: str,
    output_dir: str = ".",
):
    api = NativeApi(dataverse_baseurl)
    data_api = DataAccessApi(dataverse_baseurl)
    dataset = api.get_dataset(dataverse_pid)
    files_list = dataset.json()["data"]["latestVersion"]["files"]
    file2id = {f["dataFile"]["filename"]: f["dataFile"]["id"] for f in files_list}

    if filename not in file2id:
        raise ValueError(f"File {filename} not found in dataverse.")
    else:
        response = data_api.get_datafile(file2id[filename])
        with open(f"{output_dir}/{filename}", "wb") as f:
            f.write(response.content)


def upload_dataverse_data(
    data_path: str,
    data_description: str,
    dataverse_baseurl: str,
    dataverse_pid: str,
    dataverse_token: str,
    dataset_publish: bool = False,
    debug: bool = False,
):
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
    dataverse_datafile.set(
        {
            "pid": dataverse_pid,
            "filename": filename,
            "description": data_description,
        }
    )
    LOGGER.info("File basename: " + filename)

    if not debug:
        resp = api.upload_datafile(dataverse_pid, data_path, dataverse_datafile.json())
        if resp.json()["status"] == "OK":
            LOGGER.info("Dataset uploaded.")
        else:
            LOGGER.error("Dataset not uploaded.")
            LOGGER.error(resp.json())

        if dataset_publish:
            resp = api.publish_dataset(dataverse_pid, release_type="major")
            if resp.json()["status"] == "OK":
                LOGGER.info("Dataset published.")

    else:
        LOGGER.info("Debug mode. Dataset not uploaded.")


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

    LOGGER.info(f"Best lambda: {best_lam:.4f}")
    losses_ = {np.round(k, 4): np.round(v, 4) for k, v in losses.items()}
    LOGGER.info(f"Losses: {losses_}")

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


def generate_noise_like(x: pd.Series, graph: nx.Graph) -> np.ndarray:
    """Injects noise into residuals using a Gaussian Markov Random Field."""
    # find average correlation between a point and its neighbors mean
    nbrs2ix = {n: i for i, n in enumerate(x.index)}
    nbrs = [[nbrs2ix[i] for i in nx.neighbors(graph, n)] for n in x.index]
    x_ = x.values
    nbr_means = np.array([np.nanmean(x_[nbrs[i]]) for i in range(len(x))])
    x__ = (x_ - np.nanmean(x_)) / np.nanstd(x_)
    nbr_means_ = (nbr_means - np.nanmean(nbr_means)) / np.nanstd(nbr_means)
    corr = np.nansum(x__ * nbr_means_) / len(x)

    # scaled graph laplacian
    Q = nx.laplacian_matrix(graph).toarray() / np.mean([len(n) for n in nbrs])

    # add variance to diagonal
    Q = corr * Q + (1 - corr + 1e-4) * np.eye(Q.shape[0])
    LOGGER.info(f"Residual neighbor correlation: {corr:.4f}")

    # sample from GMRF
    Z = np.random.randn(Q.shape[0])
    L = cholesky(Q, lower=True)
    noise = solve_triangular(L, Z, lower=True).T

    # scale noise to have same variance as residuals
    noise = noise / noise.std() * np.nanstd(x)

    return noise


def moran_I(x: pd.Series, A: np.ndarray) -> float:
    x = x.values

    # input the mean of x when there nan values
    x[np.isnan(x)] = np.nanmean(x)

    # Compute mean of attribute values
    x_bar = np.mean(x)

    # Subtract mean from attribute values
    x_diff = x - x_bar

    # Compute denominator: sum of squared differences from mean
    denominator = np.sum(x_diff**2) + 1e-8

    # Compute numerator: sum of product of weight and pair differences from mean
    # Matrix multiplication to achieve vectorization
    numerator = np.sum(A * np.outer(x_diff, x_diff))

    # Compute Moran's I
    I = len(x) / np.sum(A) * (numerator / denominator)

    return float(I)


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
