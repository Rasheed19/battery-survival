import logging
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from itertools import islice
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml
from scipy.interpolate import interp1d
from sklearn.pipeline import Pipeline
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

from utils.definitions import CIMode, Definition


def get_rcparams():
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 11

    FONT = {"family": "serif", "serif": ["Times"], "size": MEDIUM_SIZE}

    rc_params = {
        "axes.titlesize": MEDIUM_SIZE,
        "axes.labelsize": SMALL_SIZE,
        "xtick.labelsize": SMALL_SIZE,
        "ytick.labelsize": SMALL_SIZE,
        "legend.fontsize": SMALL_SIZE,
        "figure.titlesize": BIGGER_SIZE,
        "font.family": FONT["family"],
        "font.serif": FONT["serif"],
        "font.size": FONT["size"],
        "text.usetex": True,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "xtick.bottom": True,
        "ytick.left": True,
        "figure.constrained_layout.use": True,
    }

    return rc_params


def time_monitor(initial_time: datetime | None = None) -> str:
    """
    This function monitors time from the start of a process to the end of the process
    """
    if not initial_time:
        initial_time = datetime.now()
        return initial_time
    else:
        thour, temp_sec = divmod((datetime.now() - initial_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)

        return "%ih %imin and %ss." % (thour, tmin, round(tsec, 2))


def read_data(fname: str, path: str) -> Any:
    """
    Function that reads .pkl file from a
    a given folder.

    Args:
    ----
        fname (str):  file name
        path (str): path to folder

    Returns:
    -------
            loaded file.
    """
    # Load pickle data
    with open(os.path.join(path, fname), "rb") as fp:
        loaded_file = pickle.load(fp)

    return loaded_file


def dump_data(data: Any, fname: str, path: str) -> None:
    """
    Function that dumps a pickled data into
    a specified path

    Args:
    ----
        data (Any): data to be pickled
        fname (str):  file name
        path (str): path to folder

    Returns:
    -------
            None
    """
    with open(os.path.join(path, fname), "wb") as fp:
        pickle.dump(data, fp)

    return None


def load_yaml_file(path: str) -> dict[Any, Any]:
    with open(path, "r") as file:
        data = yaml.safe_load(file)

    return data


def check_cells_stats(
    batches: list, loaded_cells: list, loaded_data: dict
) -> pd.DataFrame:
    cell_stats = pd.DataFrame(index=batches)
    num_of_cells = []

    for b in batches:
        num_of_cells.append(
            len(
                [
                    cell
                    for cell in loaded_cells
                    if loaded_data[cell]["summary_data"]["batch_name"] == b
                ]
            )
        )

    cell_stats["Number of cells"] = num_of_cells
    cell_stats.index.name = "Batches"

    return cell_stats


def score_survival_model(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> float:
    prediction = model.predict(X)
    result = concordance_index_censored(
        y["cycled_to_eol"], y["end_of_life"], prediction
    )
    return result[0]


class CustomFormatter(logging.Formatter):
    purple = "\x1b[1;35m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(levelname)s: [%(name)s] %(message)s"
    # "[%(asctime)s] (%(name)s) %(levelname)s: %(message)s"
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: purple + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(root_logger: str) -> logging.Logger:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler],
    )

    return logging.getLogger(root_logger)


def get_dict_subset(dictionary: dict, length: int) -> dict:
    return dict(islice(dictionary.items(), length))


@dataclass(frozen=True)
class SurvivalModelEvaluationMetrics:
    c_index: float
    time_dependent_auc: float
    time_dependent_brier_score: float


def get_survival_metrics(
    model: Pipeline,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    times: np.ndarray,
) -> SurvivalModelEvaluationMetrics:
    time_independent_risk = model.predict(X_test)
    c_index = concordance_index_censored(
        y_test["cycled_to_eol"], y_test["end_of_life"], time_independent_risk
    )[0]

    _, time_dependent_auc = cumulative_dynamic_auc(
        survival_train=y_train,
        survival_test=y_test,
        estimate=time_independent_risk,
        times=times,
    )

    surv_funcs = model.predict_survival_function(X_test, return_array=False)
    surv_probs = np.row_stack([fn(times) for fn in surv_funcs])
    time_dependent_brier_score = integrated_brier_score(
        survival_train=y_train,
        survival_test=y_test,
        estimate=surv_probs,
        times=times,
    )

    return SurvivalModelEvaluationMetrics(
        c_index=c_index,
        time_dependent_auc=time_dependent_auc,
        time_dependent_brier_score=time_dependent_brier_score,
    )


@dataclass(frozen=True)
class BootstrapSurvivalMetrics:
    c_index_bootstrap: np.ndarray
    time_dependent_auc_bootstrap: np.ndarray
    time_dependent_brier_bootstrap: np.ndarray


def bootstrap_survival_metrics(
    n_bootstraps: int,
    model: Pipeline,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    times: np.ndarray | None = None,
) -> np.ndarray:
    bootstraps = np.zeros(shape=(n_bootstraps, 3))
    sample_size = X_test.shape[0]

    for i in range(n_bootstraps):
        sample_idx_ = np.random.randint(
            low=0, high=sample_size, size=sample_size
        )  # end value is excluded

        metrics = get_survival_metrics(
            model=model,
            y_train=y_train,
            X_test=X_test.iloc[sample_idx_],
            y_test=y_test[sample_idx_],
            times=times,
        )

        for j, value in enumerate(
            [
                metrics.c_index,
                metrics.time_dependent_auc,
                metrics.time_dependent_brier_score,
            ]
        ):
            bootstraps[i, j] = value

    return BootstrapSurvivalMetrics(
        c_index_bootstrap=bootstraps[:, 0],
        time_dependent_auc_bootstrap=bootstraps[:, 1],
        time_dependent_brier_bootstrap=bootstraps[:, 2],
    )


def empirical_cdf(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the empirical cumulative distribution function (CDF)
    from an array of data points.

    Args:
    ----
         data: the input data.

    Returns:
    -------
            x (numpy array): The unique data points sorted in ascending order.
            cdf (numpy array): The corresponding CDF values for each data point.
    """
    # Sort the data in ascending order
    sorted_data = np.sort(data)

    # Calculate the unique data points and their counts
    unique_data, counts = np.unique(sorted_data, return_counts=True)

    # Calculate the CDF values
    cdf = np.cumsum(counts) / len(data)

    return unique_data, cdf


def inverse_empirical_cdf(data: np.ndarray, probability: float) -> float:
    """
    Calculate the inverse empirical cumulative distribution
    function (CDF) from a list of data points.

    Args:
    ----
        data: the input data.

    Returns:
    -------
            inverse cdf corresponding to the probability
    """

    # Ensure the probability is within [0, 1]
    if 0 > probability > 1:
        raise ValueError("probabilty must be in the interval [0, 1]")

    # Calculate the empirical CDF
    x, cdf = empirical_cdf(data)

    # Create an interpolating function for the CDF
    cdf_interpolated = interp1d(
        cdf, x, kind="linear", fill_value=(x[0], x[-1]), bounds_error=False
    )

    # Use the inverse CDF function to find the corresponding value
    return cdf_interpolated(probability)


def pivotal_confidence_interval(
    estimate_on_all_samples: float, bootstraps: np.ndarray, alpha: float
) -> tuple[float, float]:
    bootstraps_shifted = bootstraps - estimate_on_all_samples
    return (
        estimate_on_all_samples
        - inverse_empirical_cdf(bootstraps_shifted, 1 - (alpha / 2)),
        estimate_on_all_samples - inverse_empirical_cdf(bootstraps_shifted, alpha / 2),
    )


def percentile_confidence_interval(
    data: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    alpha_tail = alpha / 2
    data_sorted = np.sort(data)
    n_bootstraps = len(data_sorted)

    return (
        data_sorted[int(alpha_tail * n_bootstraps)],
        data_sorted[int((1 - alpha_tail) * n_bootstraps)],
    )


@dataclass(frozen=True)
class SurvivalModelMetricCI:
    c_index_ci: tuple[float, float]
    time_dependent_auc_ci: tuple[float, float]
    time_dependent_brier_score_ci: tuple[float, float]


def get_survival_metric_ci(
    bootstrap_survival_metrics: BootstrapSurvivalMetrics,
    metric_estimates: SurvivalModelEvaluationMetrics,
    alpha: float,
    interval_type: str = CIMode.PERCENTILE,
) -> SurvivalModelMetricCI:
    if interval_type not in CIMode:
        raise ValueError(
            f"'interval_type' must be {CIMode.PERCENTILE} or {CIMode.PIVOTAL} "
            f"but {interval_type} is given."
        )

    ci: list[tuple] = []

    for b, e in zip(
        [
            bootstrap_survival_metrics.c_index_bootstrap,
            bootstrap_survival_metrics.time_dependent_auc_bootstrap,
            bootstrap_survival_metrics.time_dependent_brier_bootstrap,
        ],
        [
            metric_estimates.c_index,
            metric_estimates.time_dependent_auc,
            metric_estimates.time_dependent_brier_score,
        ],
    ):
        if interval_type == CIMode.PERCENTILE:
            ci.append(percentile_confidence_interval(data=b, alpha=alpha))

        else:
            ci.append(
                pivotal_confidence_interval(
                    estimate_on_all_samples=e,
                    bootstraps=b,
                    alpha=alpha,
                )
            )

    return SurvivalModelMetricCI(
        c_index_ci=ci[0],
        time_dependent_auc_ci=ci[1],
        time_dependent_brier_score_ci=ci[2],
    )


def download_file(url: str, file_name: str, destination_folder: str = "data") -> None:
    response = requests.get(url)
    with open(f"{Definition.ROOT_DIR}/{destination_folder}/{file_name}", "wb") as file:
        file.write(response.content)
