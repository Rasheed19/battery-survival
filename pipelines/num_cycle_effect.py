import numpy as np

from steps import (
    data_loader,
    data_modeller,
    data_splitter,
    model_trainer,
)
from utils.generic_helper import dump_data, get_logger
from utils.plotter import plot_num_cycle_effect_history

logger = get_logger(__name__)


def num_cycle_effect_pipeline(
    loaded_cycles: int,
    not_loaded: bool,
    test_size: float,
    signature_depth: int,
    parameter_space: dict,
) -> None:
    logger.info("Signature effect pipeline has started.")

    # load data
    loaded_data = data_loader(loaded_cycles=loaded_cycles, not_loaded=not_loaded)

    # split data
    split_data = data_splitter(
        loaded_data=loaded_data,
        test_size=test_size,
    )

    MIN_CYCLE, MAX_CYCLE = 3, 100

    cycle_number_list = np.arange(MIN_CYCLE, MAX_CYCLE + 1)
    history = {}

    for regime in ["charge", "discharge"]:
        cv_scores = []

        for num_cycles in cycle_number_list:
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
            _, best_cv_score = model_trainer(
                X_train=data_modeller_output.X_train,
                y_train=data_modeller_output.y_train,
                parameter_space=parameter_space,
            )

            cv_scores.append(best_cv_score)

            print(
                f"regime={regime}, num-cycles={num_cycles}/{MAX_CYCLE}, cv c-index={best_cv_score:.4f}"
            )

        history[regime] = cv_scores

    history["cycle_number_list"] = cycle_number_list
    dump_data(
        data=history,
        fname="num_cycle_effect_history.pkl",
        path="./data",
    )
    # history = read_data(
    #     fname="num_cycle_effect_history.pkl",
    #     path="./assets",
    # )
    plot_num_cycle_effect_history(history=history)

    logger.info("Signature effect pipeline finished successfully.")

    return None
