from utils.data_wrangler import get_modelled_data, DataModeller


def data_modeller(
    loaded_data: dict,
    num_cycles: int,
    regime: str,
    train_cells: list,
    test_cells: list,
    signature_depth: int,
) -> DataModeller:

    X_train, y_train = get_modelled_data(
        data=loaded_data,
        regime=regime,
        num_cycles=num_cycles,
        cell_list=train_cells,
        signature_depth=signature_depth,
    )
    X_test, y_test = get_modelled_data(
        data=loaded_data,
        regime=regime,
        num_cycles=num_cycles,
        cell_list=test_cells,
        signature_depth=signature_depth,
    )

    return DataModeller(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
