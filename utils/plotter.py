import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from utils.definitions import Definition
from utils.generic_helper import (
    get_rcparams,
    read_data,
    score_survival_model,
)

plt.rcParams.update(get_rcparams())


def set_size(
    width: float | str = 360.0,
    fraction: float = 1.0,
    subplots: tuple = (1, 1),
    adjust_height: float | None = None,
) -> tuple:
    # for general use
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    if adjust_height is not None:
        golden_ratio += adjust_height

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def plot_eol_strip_plot(
    all_batch_data: dict,
    unique_batch_labels: list,
    save_tag: str,
) -> None:
    _, ax = plt.subplots(figsize=set_size())

    batch_labels = []
    end_of_life = []
    cycled_to_eol = []

    for label in unique_batch_labels:
        temp_batch = {
            cell: all_batch_data[cell]
            for cell in all_batch_data
            if all_batch_data[cell]["summary_data"]["batch_name"] == label
        }
        temp_end_of_life = [
            temp_batch[cell]["summary_data"]["end_of_life"] for cell in temp_batch
        ]
        end_of_life.extend(temp_end_of_life)
        batch_labels.extend([label] * len(temp_end_of_life))

        temp_cycled_to_eol = []
        for cell in temp_batch:
            censored = cell[:2] in ["b4", "b5", "b6", "b7"]
            temp_cycled_to_eol.append("Censored" if censored else "Uncensored")
        cycled_to_eol.extend(temp_cycled_to_eol)

    batch_label_eol_df = pd.DataFrame()
    batch_label_eol_df["End of life"] = end_of_life
    batch_label_eol_df["Batch"] = batch_labels
    batch_label_eol_df["Censorship"] = cycled_to_eol

    print(
        f"max, min eol = {batch_label_eol_df['End of life'].min()}, {batch_label_eol_df['End of life'].max()}"
    )

    sns.stripplot(
        data=batch_label_eol_df,
        x="End of life",
        y="Batch",
        hue="Censorship",
        alpha=0.6,
        hue_order=["Censored", "Uncensored"],
        marker="o",
        ax=ax,
        linewidth=1,
        palette=["crimson", "darkcyan"],
    )

    plt.savefig(
        f"{Definition.ROOT_DIR}/plots/surv_proj_{save_tag}.pdf",
        bbox_inches="tight",
    )

    return None


def plot_voltage_curve_by_batch(
    loaded_data: dict, num_cycles: int, regime: str
) -> None:
    alphabet_tags = ["a", "b", "c", "d", "e", "f", "g", "h"]
    fig = plt.figure(figsize=set_size(subplots=(2, 4), adjust_height=0.1))

    for i, batch in enumerate(Definition.TOYOTA_BATCHES):
        ax = fig.add_subplot(2, 4, i + 1)
        ax.text(
            x=-0.1,
            y=1.4,
            s=r"\bf {}".format(alphabet_tags[i]),
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )

        batch_data = {
            cell: loaded_data[cell]
            for cell in loaded_data
            if loaded_data[cell]["summary_data"]["batch_name"] == batch
        }
        for d in batch_data.values():
            time, voltage = d["cycle_data"][str(num_cycles)][regime]
            ax.plot(
                time,
                voltage,
                linewidth=0.1,
                color="darkcyan" if regime == "charge" else "crimson",
            )

        if i in [4, 5, 6, 7]:
            ax.set_xlabel("Time (min)")

        if i % 4 == 0:
            ax.set_ylabel("Voltage (V)")

        ax.xaxis.set_major_locator(tck.MaxNLocator(nbins=4, steps=[5]))
        ax.yaxis.set_major_locator(tck.MaxNLocator(nbins=4))

    plt.savefig(
        f"{Definition.ROOT_DIR}/plots/surv_proj_voltage_curve_by_batch_{regime}.pdf",
        bbox_inches="tight",
    )

    return None


def plot_data_increment_effect_history(
    history: dict[str, list[float] | np.ndarray],
) -> None:
    fig = plt.figure(figsize=set_size(subplots=(1, 3)))
    ylabels = ["C-index", "Cum. dynamic AUC", "Int. Brier score"]
    alphabet_tags = ["a", "b", "c"]

    for i, ylabel in enumerate(ylabels):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.text(
            x=-0.1,
            y=1.25,
            s=r"\bf {}".format(alphabet_tags[i]),
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )
        ax.plot(
            history["sample_fractions"],
            history["stacked_metrics_cg"][:, i],
            label="charge",
            color="darkcyan",
            linestyle="-",
        )
        ax.plot(
            history["sample_fractions"],
            history["stacked_metrics_dg"][:, i],
            label="discharge",
            color="crimson",
            linestyle="--",
        )

        ax.xaxis.set_major_locator(tck.MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(tck.MaxNLocator(nbins=4))
        ax.set_xlabel("Sample fraction")
        ax.set_ylabel(ylabel)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.3))

    plt.savefig(
        f"{Definition.ROOT_DIR}/plots/surv_proj_data_increment_effect.pdf",
        bbox_inches="tight",
    )

    return None


def plot_num_cycle_effect_history(history: dict[str, list[float] | np.ndarray]) -> None:
    regimes = ["charge", "discharge"]
    colors = ["darkcyan", "crimson"]
    line_styles = ["-", "--"]
    fig, ax = plt.subplots(figsize=set_size())

    for i, r in enumerate(regimes):
        ax.plot(
            history["cycle_number_list"],
            history[r],
            label=r,
            color=colors[i],
            linestyle=line_styles[i],
        )

    ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True, steps=[2, 10]))
    ax.yaxis.set_major_locator(tck.MaxNLocator(integer=True, steps=[1, 3]))
    ax.set_xlabel("Cycle number threshold")
    ax.set_ylabel("Cross-validated C-index")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))

    plt.savefig(
        f"{Definition.ROOT_DIR}/plots/surv_proj_num_cycle_effect.pdf",
        bbox_inches="tight",
    )

    return None


def plot_sig_effect_history(history: dict[str, list[float] | np.ndarray]) -> None:
    regimes = ["charge", "discharge"]
    colors = ["darkcyan", "crimson"]
    line_styles = ["-", "--"]
    fig, ax = plt.subplots(figsize=set_size())

    for i, r in enumerate(regimes):
        ax.plot(
            history["signature_depths"],
            history[r],
            label=r,
            color=colors[i],
            linestyle=line_styles[i],
        )

    ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
    ax.set_xlabel("Signature depth")
    ax.set_ylabel("Cross-validated C-index")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))

    plt.savefig(
        f"{Definition.ROOT_DIR}/plots/surv_proj_sig_effect.pdf",
        bbox_inches="tight",
    )

    return None


def plot_low_cycle_prediction_history(
    history: dict[str, list[float] | np.ndarray],
) -> None:
    fig = plt.figure(figsize=set_size(subplots=(1, 3)))
    ylabels = ["C-index", "Cum. dynamic AUC", "Int. Brier score"]
    alphabet_tags = ["a", "b", "c"]

    for i, ylabel in enumerate(ylabels):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.text(
            x=-0.1,
            y=1.25,
            s=r"\bf {}".format(alphabet_tags[i]),
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )
        ax.plot(
            history["cycle_number_list"],
            history["charge"][:, i],
            label="charge",
            color="darkcyan",
            linestyle="-",
        )
        ax.plot(
            history["cycle_number_list"],
            history["discharge"][:, i],
            label="discharge",
            color="crimson",
            linestyle="--",
        )

        ax.xaxis.set_major_locator(tck.MaxNLocator(steps=[1, 5]))
        ax.yaxis.set_major_locator(tck.MaxNLocator(nbins=4))
        ax.set_xlabel("Cycle number threshold")
        ax.set_ylabel(ylabel)

        # ax.set_xlim([None, history["cycle_number_list"].max()])
        # ax.set_ylim(
        #     [None, max(history["charge"][:, i].max(), history["discharge"][:, i].max())]
        # )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.3))

    plt.savefig(
        f"{Definition.ROOT_DIR}/plots/surv_proj_low_cycle_prediction.pdf",
        bbox_inches="tight",
    )

    return None


def plot_sparsity_robustness_history(
    history: dict[str, list[float] | np.ndarray],
    figure_tag: str,
    alphabet_tags: list[str],
) -> None:
    fig = plt.figure(figsize=set_size(subplots=(1, 3)))
    ylabels = ["C-index", "Cum. dynamic AUC", "Int. Brier score"]

    for i, ylabel in enumerate(ylabels):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.text(
            x=-0.1,
            y=1.25,
            s=r"\bf {}".format(alphabet_tags[i]),
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )
        ax.plot(
            history["step_number_array"],
            history["charge"][:, i],
            label="charge",
            color="darkcyan",
            linestyle="-",
        )
        ax.plot(
            history["step_number_array"],
            history["discharge"][:, i],
            label="discharge",
            color="crimson",
            linestyle="--",
        )

        ax.xaxis.set_major_locator(tck.MaxNLocator(steps=[1, 2]))
        ax.yaxis.set_major_locator(tck.MaxNLocator(nbins=4))
        ax.set_xlabel("Step number")
        ax.set_ylabel(ylabel)

    if figure_tag == "test":
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.3)
        )

    plt.savefig(
        f"{Definition.ROOT_DIR}/plots/surv_proj_sparsity_robustness_{figure_tag}.pdf",
        bbox_inches="tight",
    )

    return None


def plot_survival_hazard_function(
    model: Pipeline,
    loaded_data: dict[str, dict],
    regime: str,
    batch_names: list[str],
    cell_list: list[str],
    X_inf: pd.DataFrame,
    plot_type: str,
) -> None:
    # match cell to its predictions
    if plot_type == "survival":
        survival_functions = model.predict_survival_function(X_inf, return_array=True)
        ylabel = r"$\hat{S}(t)$"
    elif plot_type == "hazard":
        survival_functions = model.predict_cumulative_hazard_function(
            X_inf, return_array=True
        )
        ylabel = r"$\hat{H}(t)$"
    else:
        raise ValueError(
            f"""Wrong option. Valid options are 'survival' and 'hazard',
            but {plot_type} was provided.
            """
        )

    cell_surv_fn = {cell: surv for cell, surv in zip(cell_list, survival_functions)}

    figure_labels = ["a", "b", "c", "d", "e", "f", "g", "i"]
    subplots = (2, 4)

    fig = plt.figure(figsize=set_size(subplots=subplots, adjust_height=0.1))
    for i, batch in enumerate(batch_names):
        ax = fig.add_subplot(subplots[0], subplots[1], i + 1)
        ax.text(
            x=-0.1,
            y=1.4,
            s=r"\bf {}".format(figure_labels[i]),
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )

        temp_surv_fn = {
            cell: cell_surv_fn[cell]
            for cell in cell_list
            if loaded_data[cell]["summary_data"]["batch_name"] == batch
        }
        for cell, fn in temp_surv_fn.items():
            ax.step(
                model.named_steps["sksurv"].unique_times_,
                (
                    fn
                    if plot_type == "survival"
                    else np.log1p(
                        fn
                    )  # use log(1+x) to see the differences in hazard clearly
                ),
                where="post",
                label=cell,
                linewidth=0.2,
                color="darkcyan",
                alpha=0.5,
            )

        if i in [4, 5, 6, 7]:
            ax.set_xlabel("Cycle, $t$")

        if i % 4 == 0:
            ax.set_ylabel(ylabel)

    plt.savefig(
        f"{Definition.ROOT_DIR}/plots/surv_proj_{plot_type}_{regime}.pdf",
        bbox_inches="tight",
    )

    return None


def plot_feature_importance(
    model: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    regime: str,
) -> None:
    result = permutation_importance(
        estimator=model,
        X=X,
        y=y,
        scoring=score_survival_model,
        n_repeats=100,
        random_state=42,
    )

    result_df = pd.DataFrame(
        {
            k: abs(result[k])
            for k in (
                "importances_mean",
                "importances_std",
            )
        },
        index=X.columns,
    ).sort_values(by="importances_mean", ascending=False)

    print(f"Feature importance:\n {result_df}")

    _, ax = plt.subplots(figsize=set_size())

    ax.bar(
        result_df.index,
        result_df["importances_mean"].abs(),
        color="darkcyan",
        ec="black",
        alpha=0.75,
    )
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel("Drop in C-index")

    plt.savefig(
        fname=f"{Definition.ROOT_DIR}/plots/surv_proj_feature_importance_{regime}.pdf",
        bbox_inches="tight",
    )

    return None


def plot_sig_num_cycle_effect_history() -> None:
    sig_effect_history = read_data(
        fname="sig_effect_history.pkl",
        path="./data",
    )
    num_cycle_effect_history = read_data(
        fname="num_cycle_effect_history.pkl",
        path="./data",
    )

    regimes = ["charge", "discharge"]
    colors = ["darkcyan", "crimson"]
    line_styles = ["-", "--"]
    alphabet_tags = ["a", "b"]

    fig = plt.figure(figsize=set_size(subplots=(1, 2)))

    for i, history in enumerate([sig_effect_history, num_cycle_effect_history]):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.text(
            x=-0.1,
            y=1.2,
            s=r"\bf {}".format(alphabet_tags[i]),
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )

        if i == 0:
            for j, r in enumerate(regimes):
                ax.plot(
                    history["signature_depths"],
                    history[r],
                    label=r,
                    color=colors[j],
                    linestyle=line_styles[j],
                )

            ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(tck.MaxNLocator(nbins=5))
            ax.set_xlabel("Signature depth")
            ax.set_ylabel("Cross-validated C-index")

        else:
            for j, r in enumerate(regimes):
                ax.plot(
                    history["cycle_number_list"],
                    history[r],
                    label=r,
                    color=colors[j],
                    linestyle=line_styles[j],
                )

            ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True, steps=[2, 10]))
            ax.yaxis.set_major_locator(tck.MaxNLocator(nbins=5))
            ax.set_xlabel("Cycle number threshold")
            ax.set_ylabel("Cross-validated C-index")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.2))

    plt.savefig(
        f"{Definition.ROOT_DIR}/plots/surv_proj_sig_num_cycle_effect.pdf",
        bbox_inches="tight",
    )

    return None


if __name__ == "__main__":
    plot_sig_num_cycle_effect_history()
