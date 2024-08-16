from utils.data_wrangler import get_train_test_split
from utils.definitions import Definition


def data_splitter(
    loaded_data: dict[str, dict],
    test_size: float = 0.2,
) -> dict[str, dict]:
    """
    Split cells into training and test cells; note that
    the split is designed such that the batch proportion
    is preserved in each split.
    """

    train_cells, test_cells = get_train_test_split(
        loaded_data=loaded_data,
        batch_names=Definition.TOYOTA_BATCHES,
        test_ratio=test_size,
    )

    return {"train_cells": train_cells, "test_cells": test_cells}
