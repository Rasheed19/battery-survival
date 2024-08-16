import click

from pipelines import (
    data_increment_effect_pipeline,
    download_pipeline,
    eda_pipeline,
    num_cycle_effect_pipeline,
    sig_effect_pipeline,
    training_pipeline,
)
from utils.definitions import Definition
from utils.generic_helper import load_yaml_file


@click.command(
    help="""
    Entry point for running all pipelines.
    """
)
@click.option(
    "--pipeline",
    type=click.STRING,
    help="""
Give the name of the pipeline to run.
Valid option must be an element of
['eda', 'download', training', 'sig-effect', 'num-cycle-effect', 'data-increment-effect']
""",
)
@click.option(
    "--regime",
    type=click.STRING,
    help="""
Give the name of the data regime to use.
Valid option must be an element of
['charge', 'discharge'].
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
    of the cells in the test data.
        """,
)
def main(
    pipeline: str,
    regime: str,
    not_loaded: bool = False,
    include_inference: bool = False,
) -> None:
    MODEL_CONFIG = load_yaml_file(
        path=f"{Definition.ROOT_DIR}/config/model_config.yaml"
    )
    DATA_CONFIG = load_yaml_file(path=f"{Definition.ROOT_DIR}/config/data_config.yaml")

    if pipeline == "download":
        download_pipeline()

    elif pipeline == "eda":
        eda_pipeline(
            num_cycles=DATA_CONFIG["num_cycles"],
            loaded_cycles=DATA_CONFIG["loaded_cycles"],
            not_loaded=not_loaded,
        )

    elif pipeline == "training":
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

    elif pipeline == "sig-effect":
        sig_effect_pipeline(
            loaded_cycles=DATA_CONFIG["loaded_cycles"],
            num_cycles=DATA_CONFIG["num_cycles"],
            not_loaded=not_loaded,
            test_size=MODEL_CONFIG["test_size"],
            parameter_space=MODEL_CONFIG["parameter_space"],
        )

    elif pipeline == "num-cycle-effect":
        num_cycle_effect_pipeline(
            loaded_cycles=DATA_CONFIG["loaded_cycles"],
            not_loaded=not_loaded,
            test_size=MODEL_CONFIG["test_size"],
            signature_depth=DATA_CONFIG["signature_depth"],
            parameter_space=MODEL_CONFIG["parameter_space"],
        )

    elif pipeline == "data-increment-effect":
        data_increment_effect_pipeline(
            loaded_cycles=DATA_CONFIG["loaded_cycles"],
            num_cycles=DATA_CONFIG["num_cycles"],
            not_loaded=not_loaded,
            test_size=MODEL_CONFIG["test_size"],
            signature_depth=DATA_CONFIG["signature_depth"],
            parameter_space=MODEL_CONFIG["parameter_space"],
        )
    else:
        raise ValueError(
            f"pipeline must be an element of ['eda', 'download', training', 'sig-effect', 'num-cycle-effect', "
            f"'data-increment-effect'] but {pipeline} is given"
        )
    return None


if __name__ == "__main__":
    main()
