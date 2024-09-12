import os

import h5py
import numpy as np

from utils.definitions import DataRegime, Definition
from utils.generic_helper import dump_data, time_monitor


def load_single_batch_to_dict(
    filename: str, batch_num: str, loaded_cycles: int | None = None
) -> dict:
    """
    This function loads the downloaded matlab file into a dictionary.

    Args:
    ----
        filename:     string with the path of the data file
        batch_num:    batch number
        loaded_cycles:   number of cycles to be loaded

    Returns a dictionary with data for each cell in the batch.
    """

    # read the matlab file
    f = h5py.File(filename, "r")
    batch = f["batch"]

    # get the number of cells in this batch
    num_cells = batch["summary"].shape[0]

    # initialize a dictionary to store the result
    batch_dict = {}

    summary_features = [
        "IR",
        "QCharge",
        "QDischarge",
        "Tavg",
        "Tmin",
        "Tmax",
        "chargetime",
        "cycle",
    ]
    cycle_features = [
        "I",
        "Qc",
        "Qd",
        "Qdlin",
        "T",
        "Tdlin",
        "V",
        "discharge_dQdV",
        "t",
    ]

    for i in range(num_cells):
        # decide how many cycles will be loaded
        if loaded_cycles is None:
            loaded_cycles = f[batch["cycles"][i, 0]]["I"].shape[0]
        else:
            loaded_cycles = min(loaded_cycles, f[batch["cycles"][i, 0]]["I"].shape[0])

        if i % 10 == 0:
            print(f"* {i} cells loaded ({loaded_cycles} cycles)")

        # initialise a dictionary for this cell
        cell_dict = {
            "cycle_life": (
                f[batch["cycle_life"][i, 0]][()]
                if batch_num != 3
                else f[batch["cycle_life"][i, 0]][()] + 1
            ),
            "charge_policy": f[batch["policy_readable"][i, 0]][()]
            .tobytes()[::2]
            .decode(),
            "summary": {},
        }

        for feature in summary_features:
            cell_dict["summary"][feature] = np.hstack(
                f[batch["summary"][i, 0]][feature][0, :].tolist()
            )

        # for the cycle data
        cell_dict["cycle_dict"] = {}

        for j in range(loaded_cycles):
            cell_dict["cycle_dict"][str(j + 1)] = {}
            for feature in cycle_features:
                cell_dict["cycle_dict"][str(j + 1)][feature] = np.hstack(
                    (f[f[batch["cycles"][i, 0]][feature][j, 0]][()])
                )

        # converge into the batch dictionary
        batch_dict[f"b{batch_num}c{i}"] = cell_dict

    return batch_dict


def load_all_batches_to_dict(loaded_cycles: int | None = None) -> dict[str, dict]:
    """
    This function load downloaded matlab files as pickle files.
    Note that the battery data (downloaded from https://data.matr.io/1/) must be
    put in "data/toyota" directory. After calling this function, extracted files in
    dict will be returned.

    Args:
    ----
         loaded_cycles:  number of cycles to load

    Returns:
    -------
        all loaded batches in dict
    """

    # paths for data file with each batch of cells
    mat_filenames = {
        f"batch{b}": os.path.join(f"{Definition.ROOT_DIR}", "data/toyota", mat_file)
        for b, mat_file in zip(
            range(1, 9),
            [
                "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
                "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
                "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
                "2018-08-28_batchdata_updated_struct_errorcorrect.mat",
                "2018-09-02_batchdata_updated_struct_errorcorrect.mat",
                "2018-09-06_batchdata_updated_struct_errorcorrect.mat",
                "2018-09-10_batchdata_updated_struct_errorcorrect.mat",
                "2019-01-24_batchdata_updated_struct_errorcorrect.mat",
            ],
        )
    }

    start = time_monitor()
    print("Loading batch 1 data...")
    batch1 = load_single_batch_to_dict(
        mat_filenames["batch1"], 1, loaded_cycles=loaded_cycles
    )
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 2 data...")
    batch2 = load_single_batch_to_dict(
        mat_filenames["batch2"], 2, loaded_cycles=loaded_cycles
    )
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 3 data...")
    batch3 = load_single_batch_to_dict(
        mat_filenames["batch3"], 3, loaded_cycles=loaded_cycles
    )
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 4 data...")
    batch4 = load_single_batch_to_dict(
        mat_filenames["batch4"], 4, loaded_cycles=loaded_cycles
    )
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 5 data...")
    batch5 = load_single_batch_to_dict(
        mat_filenames["batch5"], 5, loaded_cycles=loaded_cycles
    )
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 6 data...")
    batch6 = load_single_batch_to_dict(
        mat_filenames["batch6"], 6, loaded_cycles=loaded_cycles
    )
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 7 data...")
    batch7 = load_single_batch_to_dict(
        mat_filenames["batch7"], 7, loaded_cycles=loaded_cycles
    )
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 8 data...")
    batch8 = load_single_batch_to_dict(
        mat_filenames["batch8"], 8, loaded_cycles=loaded_cycles
    )
    print(time_monitor(start))

    print(f"* {len(batch1.keys())} cells loaded in batch 1")
    print(f"* {len(batch2.keys())} cells loaded in batch 2")
    print(f"* {len(batch3.keys())} cells loaded in batch 3")
    print(f"* {len(batch4.keys())} cells loaded in batch 4")
    print(f"* {len(batch5.keys())} cells loaded in batch 5")
    print(f"* {len(batch6.keys())} cells loaded in batch 6")
    print(f"* {len(batch7.keys())} cells loaded in batch 7")
    print(f"* {len(batch8.keys())} cells loaded in batch 8")

    # there are four cells from batch1 that carried into batch2, we'll remove the data from batch2 and put it with
    # the correct cell from batch1
    b2_keys = ["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"]
    b1_keys = ["b1c0", "b1c1", "b1c2", "b1c3", "b1c4"]
    add_len = [662, 981, 1060, 208, 482]

    # append data to batch 1
    for i, bk in enumerate(b1_keys):
        batch1[bk]["cycle_life"] = batch1[bk]["cycle_life"] + add_len[i]

        for j in batch1[bk]["summary"].keys():
            if j == "cycle":
                batch1[bk]["summary"][j] = np.hstack(
                    (
                        batch1[bk]["summary"][j],
                        batch2[b2_keys[i]]["summary"][j]
                        + len(batch1[bk]["summary"][j]),
                    )
                )
            else:
                batch1[bk]["summary"][j] = np.hstack(
                    (batch1[bk]["summary"][j], batch2[b2_keys[i]]["summary"][j])
                )

        # useful when all cycles loaded
        if loaded_cycles is None:
            last_cycle = len(batch1[bk]["cycle_dict"].keys())

            for j, jk in enumerate(batch2[b2_keys[i]]["cycle_dict"].keys()):
                batch1[bk]["cycle_dict"][str(last_cycle + j)] = batch2[b2_keys[i]][
                    "cycle_dict"
                ][jk]
    """
    The authors exclude cells that:
        * were carried into batch2 but belonged to batch 1 (batch 2)
        * noisy channels (batch 3)
    """

    exc_cells = {
        "batch2": ["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"],
        "batch3": ["b3c37", "b3c2", "b3c23", "b3c32", "b3c38", "b3c39"],
    }

    for c in exc_cells["batch2"]:
        del batch2[c]

    for c in exc_cells["batch3"]:
        del batch3[c]

    # exclude the first cycle from all cells because this data was not part of the first batch of cells
    batches = [batch1, batch2, batch3, batch4, batch5, batch6, batch7, batch8]
    for batch in batches:
        for cell in batch.keys():
            del batch[cell]["cycle_dict"]["1"]

    for batch in batches:
        for cell in batch.keys():
            assert "1" not in batch[cell]["cycle_dict"].keys()

    for batch in batches:
        for cell in batch.keys():
            for feat in batch[cell]["summary"].keys():
                batch[cell]["summary"][feat] = np.delete(
                    batch[cell]["summary"][feat], 0
                )

    # combine all batches in one dictionary
    data_dict = {
        **batch1,
        **batch2,
        **batch3,
        **batch4,
        **batch5,
        **batch6,
        **batch7,
        **batch8,
    }

    return data_dict


def get_constant_indices(
    feature: list[float] | np.ndarray, regime: str
) -> tuple[int, int]:
    constant_feature_list = []
    constant_feature_index = []

    for i in range(1, len(feature)):
        if abs(feature[i - 1] - feature[i]) <= 1e-2:
            constant_feature_list.append(feature[i - 1])
            constant_feature_index.append(i - 1)

    if regime == DataRegime.CHARGE:
        det_value = np.max(constant_feature_list)
        opt_list = [
            i
            for i, element in zip(constant_feature_index, constant_feature_list)
            if np.round(det_value - element, 2) <= 0.5
        ]

        return opt_list[0], opt_list[-1]

    elif regime == DataRegime.DISCHARGE:
        det_value = np.min(constant_feature_list)
        opt_list = [
            i
            for i, element in zip(constant_feature_index, constant_feature_list)
            if np.round(element - det_value, 2) <= 0.5
        ]
        return opt_list[0], opt_list[-1]

    else:
        raise ValueError(
            f"option must be {DataRegime.CHARGE} or {DataRegime.DISCHARGE} but {regime} given."
        )


def get_charge_discharge_values(
    data_dict: dict[str, dict], col_name: str, cell: str, cycle: str, regime: str
) -> np.ndarray:
    TOL = 1e-10

    # An outlier in b1c2 at cycle 2176, measurement is in seconds and thus divide it by 60
    if cell == "b1c2" and cycle == "2176":
        summary_charge_time = (
            data_dict[cell]["summary"]["chargetime"][int(cycle) - 2] / 60
        )
    else:
        summary_charge_time = data_dict[cell]["summary"]["chargetime"][int(cycle) - 2]

    values = data_dict[cell]["cycle_dict"][cycle][col_name]

    if regime == DataRegime.CHARGE:
        return np.array(
            values[
                data_dict[cell]["cycle_dict"][cycle]["t"] - summary_charge_time <= TOL
            ]
        )
    elif regime == DataRegime.DISCHARGE:
        return np.array(
            values[
                data_dict[cell]["cycle_dict"][cycle]["t"] - summary_charge_time > TOL
            ]
        )
    else:
        raise ValueError(
            f"option must be {DataRegime.CHARGE} or {DataRegime.DISCHARGE} but {regime} given."
        )


def get_cc_voltage_curve(
    data_dict: dict[str, dict], cell: str, cycle: str, regime: str
) -> tuple[np.ndarray, np.ndarray]:
    discharge_values = {
        k: get_charge_discharge_values(data_dict, k, cell, cycle, regime)
        for k in ["I", "V", "t"]
    }

    if regime == DataRegime.CHARGE:
        ccv = discharge_values["V"]
        cct = discharge_values["t"]

        # fix outlier in cell b7c36 at cycle 50
        if cell == "b7c36" and cycle == "50":
            bool_filter = cct > 0.0
            cct = cct[bool_filter]
            ccv = ccv[bool_filter]

    elif regime == DataRegime.DISCHARGE:
        # get the indices of the start and end of CC
        start_i, end_i = get_constant_indices(discharge_values["I"], regime)

        ccv = discharge_values["V"][start_i : end_i + 1]
        cct = discharge_values["t"][start_i : end_i + 1]

    cct = cct - min(cct)

    return cct, ccv


def get_end_of_life(
    discharge_capacity: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    NOMINAL_CAPACITY = (
        1.1  # for Severson-Attia data, nominal capacity reported is 1.1 Ah
    )

    cycle = np.arange(start=1, stop=discharge_capacity.shape[0] + 1)
    end_of_life_bool = discharge_capacity >= Definition.EOL_THRESHOLD * NOMINAL_CAPACITY
    end_of_life = len(discharge_capacity[end_of_life_bool])

    return end_of_life, cycle[end_of_life_bool], discharge_capacity[end_of_life_bool]


def dump_toyota_structured_data(loaded_cycles: int) -> None:
    all_batches: dict = load_all_batches_to_dict(loaded_cycles=loaded_cycles)

    structured_data: dict = {}

    for cell, data in all_batches.items():
        cycle_data = {}

        for cycle in data["cycle_dict"]:
            cycle_data[cycle] = {
                regime.value: get_cc_voltage_curve(
                    data_dict=all_batches,
                    cell=cell,
                    cycle=cycle,
                    regime=regime.value,
                )
                for regime in DataRegime
            }

        eol, cycle_eol, discharge_capacity_eol = get_end_of_life(
            discharge_capacity=data["summary"]["QDischarge"]
        )
        summary_data = {
            "cycle": cycle_eol,
            "discharge_capacity": discharge_capacity_eol,
            "batch_name": cell[:2],
            "end_of_life": eol,
        }

        structured_data[cell] = {"cycle_data": cycle_data, "summary_data": summary_data}

    dump_data(
        data=structured_data,
        fname="toyota_data.pkl",
        path=f"{Definition.ROOT_DIR}/data",
    )

    return None
