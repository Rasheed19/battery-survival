import os
from enum import Enum, EnumMeta


class Definition:
    ROOT_DIR: str = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    TOYOTA_BATCHES: list[str] = [f"b{i}" for i in range(1, 9)]
    EOL_PROBABILITY: float = 1e-6
    ZERO_HAZARD: float = 1e-6
    EOL_THRESHOLD: float = 0.8  # end of life threshold (80% of the intial capacity)
    REPEATS: int = 100
    TIME_MIN: int = 500
    TIME_MAX: int = 1000


class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseEnum(Enum, metaclass=MetaEnum):
    pass


class PipelineMode(str, BaseEnum):
    EDA = "eda"
    DOWNLOAD = "download"
    TRAIN = "training"
    SIG_EFFECT = "sig-effect"
    CYCLE_EFFECT = "num-cycle-effect"
    INCREMENT_EFFECT = "data-increment-effect"
    LOW_CYCLE = "low-cycle-prediction"
    SPARSITY = "sparsity-robustness"


class DataRegime(str, BaseEnum):
    CHARGE = "charge"
    DISCHARGE = "discharge"


class SparsityLevel(str, BaseEnum):
    TRAIN = "train"
    TEST = "test"


class SurvivalPlot(str, BaseEnum):
    SURVIVAL = "survival"
    HAZARD = "hazard"


class CIMode(str, BaseEnum):
    PERCENTILE = "percentile"
    PIVOTAL = "pivotal"
