from steps import data_loader
from utils.definitions import Definition
from utils.generic_helper import get_logger
from utils.plotter import (
    plot_eol_strip_plot,
    plot_voltage_curve_by_batch,
)


def eda_pipeline(
    loaded_cycles: int,
    num_cycles: int,
    not_loaded: bool = False,
) -> None:
    logger = get_logger(__name__)

    logger.info("Loading combined data...")
    loaded_data = data_loader(loaded_cycles=loaded_cycles, not_loaded=not_loaded)

    logger.info("Plotting strip plot of end of life...")
    plot_eol_strip_plot(
        all_batch_data=loaded_data,
        unique_batch_labels=Definition.TOYOTA_BATCHES,
        save_tag="eol_strip_plot",
    )

    logger.info("Plotting cell voltage curve by batch...")

    for regime in ["charge", "discharge"]:
        plot_voltage_curve_by_batch(
            loaded_data=loaded_data,
            num_cycles=num_cycles,
            regime=regime,
        )

    logger.info(
        "EDA pipeline finished successfully. Check the 'plots' folder for the results."
    )

    return None
