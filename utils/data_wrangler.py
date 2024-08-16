import pandas as pd
import numpy as np
import random
import iisignature
import esig
from dataclasses import dataclass


from utils.definitions import Definition
from utils.generic_helper import load_yaml_file


def batch_splitter(
    loaded_data: dict,
    test_ratio: float,
    batch_name: str,
) -> tuple[list, list]:

    # get train ratio
    train_ratio = 1.0 - test_ratio

    cells_in_batch = [
        cell
        for cell in loaded_data
        if loaded_data[cell]["summary_data"]["batch_name"] == batch_name
    ]
    num_cells = len(cells_in_batch)

    # shuffle the cells
    random.shuffle(cells_in_batch)

    num_train_cells = int(train_ratio * num_cells)

    return cells_in_batch[:num_train_cells], cells_in_batch[num_train_cells:]


def get_train_test_split(
    loaded_data: dict, batch_names: list[str], test_ratio: float
) -> tuple[list, list]:

    train_cells, test_cells = [], []

    for batch_name in batch_names:
        tr, te = batch_splitter(
            loaded_data=loaded_data,
            batch_name=batch_name,
            test_ratio=test_ratio,
        )

        train_cells.extend(tr)
        test_cells.extend(te)

    random.shuffle(train_cells)
    random.shuffle(test_cells)

    # ensure that cell with max EOL is in the train
    train_end_of_life = [
        loaded_data[cell]["summary_data"]["end_of_life"] for cell in train_cells
    ]
    test_end_of_life = [
        loaded_data[cell]["summary_data"]["end_of_life"] for cell in test_cells
    ]

    train_argmax = np.argmax(train_end_of_life)
    test_argmax = np.argmax(test_end_of_life)

    if train_end_of_life[train_argmax] < test_end_of_life[test_argmax]:
        train_cells.append(test_cells[test_argmax])
        test_cells.remove(test_cells[test_argmax])

    return train_cells, test_cells


def get_sig_convention(dimension: int, depth: int, num_cycles: int) -> np.ndarray:
    raw_convention = esig.sigkeys(dimension=dimension, depth=depth)
    raw_convention = raw_convention.split(" ")[2:]

    sig_convention = []

    for c in ["1", str(num_cycles)]:
        for conv in raw_convention:
            conv = conv.replace("(", "{")
            conv = conv.replace(")", "}")
            sig_convention.append(f"$S^{conv}({c})$")

    return np.array(sig_convention)


def get_path_signatures(
    time: np.ndarray,
    voltage: np.ndarray,
    signature_depth: int = 2,
) -> np.ndarray:

    path = np.stack((time, voltage), axis=-1)
    return iisignature.sig(path, signature_depth)


def get_area_under_curve(x: np.ndarray, y: np.ndarray) -> float:
    return np.trapz(y=y, x=x)


def get_modelled_data(
    data: dict[str, dict],
    regime: str,
    num_cycles: int,
    cell_list: list[str],
    signature_depth: int,
) -> tuple[pd.DataFrame, np.ndarray]:

    X, y = [], []
    cycle_number_array = np.arange(2, num_cycles + 1)

    for cell in cell_list:
        signatures = []

        for c in cycle_number_array:
            time, voltage = data[cell]["cycle_data"][str(c)][regime]
            signatures.append(
                get_path_signatures(
                    time=time,
                    voltage=voltage,
                    signature_depth=signature_depth,
                ).tolist()
            )

        signatures = np.array(signatures)
        signature_of_signatures = []
        for i in range(signatures.shape[1]):
            signature_of_signatures.extend(
                get_path_signatures(
                    time=cycle_number_array,
                    voltage=signatures[:, i],  # signature of voltage as voltage
                    signature_depth=signature_depth,
                ).tolist()
            )
        X.append(signature_of_signatures)

        # # get charge voltage curves
        # t_1, v_1 = data[cell]["cycle_data"]["2"][
        #     regime
        # ]  # note cycle 2 is the first cycle as cycle 1 has been removed (originally)
        # t_num_cycles, v_num_cycles = data[cell]["cycle_data"][str(num_cycles)][regime]

        # sig_1 = get_path_signatures(
        #     time=t_1, voltage=v_1, signature_depth=signature_depth
        # )
        # sig_num_cycles = get_path_signatures(
        #     time=t_num_cycles, voltage=v_num_cycles, signature_depth=signature_depth
        # )
        # features = np.concatenate((sig_1, sig_num_cycles))

        # X.append(features.tolist())

        censored = data[cell]["summary_data"]["batch_name"] in ["b4", "b5", "b6", "b7"]
        cycled_to_eol = False if censored else True

        y.append((cycled_to_eol, data[cell]["summary_data"]["end_of_life"]))

    return pd.DataFrame(
        data=X,
        # columns=get_sig_convention(
        #     dimension=2, depth=signature_depth, num_cycles=num_cycles
        # ),
        columns=[f"feature {i + 1}" for i in range(len(X[0]))],
    ), np.array(y, dtype=[("cycled_to_eol", "?"), ("end_of_life", "<f8")])


class DataFrameCaster:
    """
    Support class to cast type back to pd.DataFrame in sklearn Pipeline.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns)


@dataclass(frozen=True)
class DataModeller:

    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
