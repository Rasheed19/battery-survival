import numpy as np

from steps import (
    data_loader,
    data_splitter,
    model_trainer,
)
from utils.data_wrangler import get_modelled_data
from utils.generic_helper import dump_data, get_logger, get_survival_metrics
from utils.plotter import plot_low_cycle_prediction_history

logger = get_logger(__name__)


def get_random_train_test_splits(
    repeats: int, loaded_data: dict, test_size: float
) -> list[dict]:
    result = []
    for _ in range(repeats):
        split_data = data_splitter(
            loaded_data=loaded_data,
            test_size=test_size,
        )
        result.append(split_data)

    return result


def low_cycle_prediction_pipeline(
    loaded_cycles: int,
    not_loaded: bool,
    num_cycles: int,
    test_size: float,
    signature_depth: int,
    parameter_space: dict,
) -> None:
    logger.info("Low cycle prediction pipeline has started.")

    loaded_data = data_loader(loaded_cycles=loaded_cycles, not_loaded=not_loaded)

    MIN_CYCLE, MAX_CYCLE = 3, 50

    cycle_number_array = np.arange(MIN_CYCLE, MAX_CYCLE + 1)

    TIME_MIN, TIME_MAX = (
        500,
        1000,
    )  # most of the cells live between these values; will be used to calculate cumm. dynamic auc
    REPEATS = 100

    history = {}

    split_data_list = get_random_train_test_splits(
        repeats=REPEATS,
        loaded_data=loaded_data,
        test_size=test_size,
    )

    for regime in ["charge", "discharge"]:
        c_index = []
        dynamic_auc = []
        brier_score = []

        for r, split_data in enumerate(split_data_list):
            X_train, y_train = get_modelled_data(
                data=loaded_data,
                regime=regime,
                num_cycles=num_cycles,
                cell_list=split_data["train_cells"],
                signature_depth=signature_depth,
            )
            model, _ = model_trainer(
                X_train=X_train,
                y_train=y_train,
                parameter_space=parameter_space,
            )

            for n in cycle_number_array:
                X_test, y_test = get_modelled_data(
                    data=loaded_data,
                    regime=regime,
                    num_cycles=n,
                    cell_list=split_data["test_cells"],
                    signature_depth=signature_depth,
                )
                metrics = get_survival_metrics(
                    model=model,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    times=np.arange(TIME_MIN, TIME_MAX),
                )

                c_index.append(metrics.c_index)
                dynamic_auc.append(metrics.time_dependent_auc)
                brier_score.append(metrics.time_dependent_brier_score)

            print(f"regime={regime}, repeat={r + 1}/{REPEATS}")

        stacked_metric = np.zeros(shape=(cycle_number_array.shape[0], 3))
        for i, lst in enumerate([c_index, dynamic_auc, brier_score]):
            stacked_metric[:, i] = (
                np.array(lst).reshape(REPEATS, cycle_number_array.shape[0]).mean(axis=0)
            )
        history[regime] = stacked_metric

    history["cycle_number_list"] = cycle_number_array
    dump_data(
        data=history,
        fname="low_cycle_prediction_history.pkl",
        path="./data",
    )
    plot_low_cycle_prediction_history(history=history)

    logger.info("Low cycle prediction pipeline finished successfuly.")

    return None
