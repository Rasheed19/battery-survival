import numpy as np

from steps import (
    data_loader,
    data_modeller,
    data_splitter,
    model_trainer,
)
from utils.definitions import DataRegime, Definition
from utils.generic_helper import dump_data, get_logger, get_survival_metrics
from utils.plotter import plot_data_increment_effect_history


def get_all_cell_samples(
    train_cells: list[str], max_eol_cell: str, repeats: int, frac: float
) -> list[list[str]]:
    all_samples = []
    for _ in range(repeats):
        sample_train_cells = []
        for batch in Definition.TOYOTA_BATCHES:
            cells_in_batch = [cell for cell in train_cells if cell[:2] == batch]
            sample_train_cells.extend(
                np.random.choice(
                    a=cells_in_batch,
                    size=int(frac * len(cells_in_batch)),
                    replace=False,
                ).tolist()
            )
        sample_train_cells.append(
            max_eol_cell
        )  # ensure the largest time is in the sampled train cells
        np.random.shuffle(sample_train_cells)

        all_samples.append(sample_train_cells)

    return all_samples


def data_increment_effect_pipeline(
    loaded_cycles: int,
    num_cycles: int,
    not_loaded: bool,
    test_size: float,
    parameter_space: dict,
    signature_depth: int,
) -> None:
    logger = get_logger(__name__)
    logger.info("Data increment effect pipeline has started.")

    loaded_data = data_loader(loaded_cycles=loaded_cycles, not_loaded=not_loaded)

    split_data = data_splitter(
        loaded_data=loaded_data,
        test_size=test_size,
    )

    sample_fractions = np.linspace(0.2, 1, 9)

    train_cells = split_data["train_cells"]
    train_eol = [
        loaded_data[cell]["summary_data"]["end_of_life"] for cell in train_cells
    ]
    max_eol_cell = train_cells[np.argmax(train_eol)]
    train_cells.remove(max_eol_cell)

    stacked_metrics_cg = []
    stacked_metrics_dg = []

    for frac in sample_fractions:
        all_cell_samples = get_all_cell_samples(
            train_cells=train_cells,
            max_eol_cell=max_eol_cell,
            repeats=Definition.REPEATS,
            frac=frac,
        )

        for regime in DataRegime:
            c_index = []
            dynamic_auc = []
            brier_score = []
            for sample_train_cells in all_cell_samples:
                data_modeller_output = data_modeller(
                    loaded_data=loaded_data,
                    num_cycles=num_cycles,
                    regime=regime.value,
                    train_cells=sample_train_cells,
                    test_cells=split_data["test_cells"],
                    signature_depth=signature_depth,
                )
                model, _ = model_trainer(
                    X_train=data_modeller_output.X_train,
                    y_train=data_modeller_output.y_train,
                    parameter_space=parameter_space,
                )
                metrics = get_survival_metrics(
                    model=model,
                    y_train=data_modeller_output.y_train,
                    X_test=data_modeller_output.X_test,
                    y_test=data_modeller_output.y_test,
                    times=np.arange(Definition.TIME_MIN, Definition.TIME_MAX),
                )
                c_index.append(metrics.c_index)
                dynamic_auc.append(metrics.time_dependent_auc)
                brier_score.append(metrics.time_dependent_brier_score)

            print(
                f"regime={regime.value}, sample frac={frac:.2f}, no. of cells={len(sample_train_cells)}, "
                f"c-index={np.mean(c_index):.4f}, int. dynamic auc={np.mean(dynamic_auc):.4f}, "
                f"int. brier score={np.mean(brier_score):.4f}"
            )

            if regime == DataRegime.CHARGE:
                stacked_metrics_cg.append(
                    [np.mean(lst) for lst in [c_index, dynamic_auc, brier_score]]
                )
            else:
                stacked_metrics_dg.append(
                    [np.mean(lst) for lst in [c_index, dynamic_auc, brier_score]]
                )

    history = {
        "sample_fractions": sample_fractions,
        "stacked_metrics_cg": np.array(stacked_metrics_cg),
        "stacked_metrics_dg": np.array(stacked_metrics_dg),
    }
    dump_data(
        data=history,
        fname="data_increment_effect_history.pkl",
        path="./data",
    )

    # history = read_data(
    #     fname="data_increment_effect_history.pkl",
    #     path="./assets",
    # )

    plot_data_increment_effect_history(
        history=history,
    )
    logger.info("Data increment effect pipeline has finished successfully.")

    return None
