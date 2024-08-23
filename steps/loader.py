from utils.definitions import Definition
from utils.generic_helper import check_cells_stats, read_data
from utils.toyota import dump_toyota_structured_data


def data_loader(
    loaded_cycles: int,
    not_loaded: bool = False,
    verbose: int = 1,
) -> dict:
    if not_loaded:
        dump_toyota_structured_data(loaded_cycles=loaded_cycles)

    # read the loaded data
    loaded_data = read_data(fname="toyota_data.pkl", path=f"{Definition.ROOT_DIR}/data")

    # check cell stats
    if verbose > 0:
        stats = check_cells_stats(
            batches=Definition.TOYOTA_BATCHES,
            loaded_cells=list(loaded_data.keys()),
            loaded_data=loaded_data,
        )

        print(f"Toyota cell stats:\n{stats}")
        print(f"Total cells: {stats['Number of cells'].sum()}")

    return loaded_data
