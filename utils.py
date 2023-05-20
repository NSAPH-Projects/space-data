from typing import Literal
import numpy as np


def scale_variable(
    x: np.ndarray, scaling: Literal["unit", "standard"] = None
) -> np.ndarray:
    """Scales a variable according to the specified scale."""
    if scaling is None:
        return x
    else:
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
    else:
        match transform:
            case "log":
                return np.log(x)
            case "symlog":
                return np.sign(x) * np.log(np.abs(x))
            case "logit":
                return np.log(x / (1 - x))
            case _:
                raise ValueError(f"Unknown transform: {transform}")
