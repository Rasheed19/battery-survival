import numpy as np

from steps import (
    data_loader,
    data_splitter,
    data_modeller,
    model_trainer,
)
from utils.generic_helper import get_logger, dump_data
from utils.plotter import plot_sig_effect_history


def sig_effect_pipeline(
    loaded_cycles: int,
    num_cycles: int,
    not_loaded: bool,
    test_size: float,
    parameter_space: dict,
) -> None:

    logger = get_logger(__name__)

    logger.info("Signature effect pipeline has started.")

    # load data
    loaded_data = data_loader(loaded_cycles=loaded_cycles, not_loaded=not_loaded)

    # split data
    split_data = data_splitter(
        loaded_data=loaded_data,
        test_size=test_size,
    )

    signature_depths = np.arange(2, 5)
    history = {}

    for regime in ["charge", "discharge"]:
        cv_scores = []

        for depth in signature_depths:

            # call data modeller
            data_modeller_output = data_modeller(
                loaded_data=loaded_data,
                num_cycles=num_cycles,
                regime=regime,
                train_cells=split_data["train_cells"],
                test_cells=split_data["test_cells"],
                signature_depth=depth,
            )

            # train model
            _, best_cv_score = model_trainer(
                X_train=data_modeller_output.X_train,
                y_train=data_modeller_output.y_train,
                parameter_space=parameter_space,
            )

            cv_scores.append(best_cv_score)

            print(
                f"regime={regime}, signature depth={depth}, cv c-index={best_cv_score:.4f}"
            )

        history[regime] = cv_scores

    history["signature_depths"] = signature_depths
    dump_data(
        data=history,
        fname="sig_effect_history.pkl",
        path="./data",
    )
    plot_sig_effect_history(history=history)

    logger.info("Signature effect pipeline finished successfully.")

    return None
