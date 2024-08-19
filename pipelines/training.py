import numpy as np
import pandas as pd

from steps import (
    data_loader,
    data_modeller,
    data_splitter,
    model_trainer,
)
from utils.definitions import Definition
from utils.generic_helper import get_logger, get_survival_metrics
from utils.plotter import plot_survival_hazard_fn


def training_pipeline(
    loaded_cycles: int,
    num_cycles: int,
    regime: str,
    not_loaded: bool,
    test_size: float,
    parameter_space: dict,
    signature_depth: int,
    include_inference: bool,
) -> None:
    logger = get_logger(__name__)

    logger.info(f"Training pipeline has started using data from {regime} regime.")

    # load data
    loaded_data = data_loader(loaded_cycles=loaded_cycles, not_loaded=not_loaded)

    REPEATS = 100
    TIME_MIN, TIME_MAX = (
        500,
        1000,
    )  # most of the cells live between these values; will be used to calculate cumm. dynamic auc

    c_index = []
    dynamic_auc = []
    brier_score = []

    for i in range(REPEATS):
        # split data
        split_data = data_splitter(
            loaded_data=loaded_data,
            test_size=test_size,
        )

        # call data modeller
        data_modeller_output = data_modeller(
            loaded_data=loaded_data,
            num_cycles=num_cycles,
            regime=regime,
            train_cells=split_data["train_cells"],
            test_cells=split_data["test_cells"],
            signature_depth=signature_depth,
        )

        # train model
        model, _ = model_trainer(
            X_train=data_modeller_output.X_train,
            y_train=data_modeller_output.y_train,
            parameter_space=parameter_space,
        )

        # evaluate model
        metrics = get_survival_metrics(
            model=model,
            y_train=data_modeller_output.y_train,
            X_test=data_modeller_output.X_test,
            y_test=data_modeller_output.y_test,
            times=np.arange(TIME_MIN, TIME_MAX),
        )

        c_index.append(metrics.c_index)
        dynamic_auc.append(metrics.time_dependent_auc)
        brier_score.append(metrics.time_dependent_brier_score)

        print(
            f"repeat {i+1}/{REPEATS}: c-index={metrics.c_index:.4f}, "
            f"dynamic-auc={metrics.time_dependent_auc:.4f}, brier-score={metrics.time_dependent_brier_score:.4f}"
        )

    logger.info("Getting training results...")

    results = pd.DataFrame(
        index=["c-index", "cum. dynamic auc", "int. brier score"],
    )
    results["mean"] = [np.mean(lst) for lst in [c_index, dynamic_auc, brier_score]]
    results["std"] = [np.std(lst) for lst in [c_index, dynamic_auc, brier_score]]
    print(results)

    if include_inference:
        logger.info(
            "Plotting survival and hazard functions with respect to source model..."
        )
        for pt in ["survival", "hazard"]:
            plot_survival_hazard_fn(
                model=model,
                loaded_data=loaded_data,
                regime=regime,
                batch_names=Definition.TOYOTA_BATCHES,
                cell_list=split_data["test_cells"],
                X_inf=data_modeller_output.X_test,
                plot_type=pt,
            )

        logger.info(
            "Inference finished successfully; check the 'plots' folder for the "
            "generated plots."
        )

    logger.info("Training pipeline finished successfully.")

    return None
