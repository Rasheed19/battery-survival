import click

from pipelines import (
    data_increment_effect_pipeline,
    download_pipeline,
    eda_pipeline,
    low_cycle_prediction_pipeline,
    num_cycle_effect_pipeline,
    sig_effect_pipeline,
    sparsity_robustness_pipeline,
    training_pipeline,
)
from utils.definitions import DataRegime, Definition, PipelineMode, SparsityLevel
from utils.generic_helper import load_yaml_file


@click.command(
    help="""
    Entry point for running all pipelines.
    """
)
@click.option(
    "--pipeline",
    type=click.STRING,
    help=f"""
Give the name of the pipeline to run.
Valid option must be an element of {[m.value for m in PipelineMode]}.
""",
)
@click.option(
    "--regime",
    type=click.STRING,
    help=f"""
Give the name of the data regime to use for running the training pipeline.
This arg must be used with the {PipelineMode.TRAIN} pipeline.
Valid option must be an element of {[r.value for r in DataRegime]}.
""",
)
@click.option(
    "--sparsity-level",
    type=click.STRING,
    help=f"""
Give where to introduce sparsity,
either in the train or test data. This
arg must be used with {PipelineMode.SPARSITY}
pipeline. Valid options are {[sl.value for sl in SparsityLevel]}.
""",
)
@click.option(
    "--not-loaded",
    is_flag=True,
    default=False,
    help="State if all batches have been loaded to a python dictionary.",
)
@click.option(
    "--include-inference",
    is_flag=True,
    default=False,
    help="""Whether to run inference on source test
    data. The inference involves predicting the
    survival and cummulative hazard functions
    of the cells in the test data. This flag must be
    used with the 'training' pipeline.
        """,
)
def main(
    pipeline: str,
    regime: str,
    sparsity_level: str,
    not_loaded: bool = False,
    include_inference: bool = False,
) -> None:
    MODEL_CONFIG = load_yaml_file(
        path=f"{Definition.ROOT_DIR}/config/model_config.yaml"
    )
    DATA_CONFIG = load_yaml_file(path=f"{Definition.ROOT_DIR}/config/data_config.yaml")

    if pipeline == PipelineMode.DOWNLOAD:
        download_pipeline()

    elif pipeline == PipelineMode.EDA:
        eda_pipeline(
            num_cycles=DATA_CONFIG["num_cycles"],
            loaded_cycles=DATA_CONFIG["loaded_cycles"],
            not_loaded=not_loaded,
        )

    elif pipeline == PipelineMode.TRAIN:
        training_pipeline(
            loaded_cycles=DATA_CONFIG["loaded_cycles"],
            num_cycles=DATA_CONFIG["num_cycles"],
            regime=regime,
            not_loaded=not_loaded,
            test_size=MODEL_CONFIG["test_size"],
            parameter_space=MODEL_CONFIG["parameter_space"],
            signature_depth=DATA_CONFIG["signature_depth"],
            include_inference=include_inference,
        )

    elif pipeline == PipelineMode.SIG_EFFECT:
        sig_effect_pipeline(
            loaded_cycles=DATA_CONFIG["loaded_cycles"],
            num_cycles=DATA_CONFIG["num_cycles"],
            not_loaded=not_loaded,
            test_size=MODEL_CONFIG["test_size"],
            parameter_space=MODEL_CONFIG["parameter_space"],
        )

    elif pipeline == PipelineMode.CYCLE_EFFECT:
        num_cycle_effect_pipeline(
            loaded_cycles=DATA_CONFIG["loaded_cycles"],
            not_loaded=not_loaded,
            test_size=MODEL_CONFIG["test_size"],
            signature_depth=DATA_CONFIG["signature_depth"],
            parameter_space=MODEL_CONFIG["parameter_space"],
        )

    elif pipeline == PipelineMode.INCREMENT_EFFECT:
        data_increment_effect_pipeline(
            loaded_cycles=DATA_CONFIG["loaded_cycles"],
            num_cycles=DATA_CONFIG["num_cycles"],
            not_loaded=not_loaded,
            test_size=MODEL_CONFIG["test_size"],
            signature_depth=DATA_CONFIG["signature_depth"],
            parameter_space=MODEL_CONFIG["parameter_space"],
        )

    elif pipeline == PipelineMode.LOW_CYCLE:
        low_cycle_prediction_pipeline(
            loaded_cycles=DATA_CONFIG["loaded_cycles"],
            num_cycles=DATA_CONFIG["num_cycles"],
            not_loaded=not_loaded,
            test_size=MODEL_CONFIG["test_size"],
            signature_depth=DATA_CONFIG["signature_depth"],
            parameter_space=MODEL_CONFIG["parameter_space"],
        )

    elif pipeline == PipelineMode.SPARSITY:
        sparsity_robustness_pipeline(
            sparsity_level=sparsity_level,
            loaded_cycles=DATA_CONFIG["loaded_cycles"],
            num_cycles=DATA_CONFIG["num_cycles"],
            not_loaded=not_loaded,
            test_size=MODEL_CONFIG["test_size"],
            signature_depth=DATA_CONFIG["signature_depth"],
            parameter_space=MODEL_CONFIG["parameter_space"],
        )
    else:
        raise ValueError(
            f"pipeline must be an element of {[m.value for m in PipelineMode]} "
            f"but {pipeline} is given"
        )
    return None


if __name__ == "__main__":
    main()
