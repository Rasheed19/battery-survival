import os


class Definition:

    ROOT_DIR: str = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    TOYOTA_BATCHES: list[str] = [f"b{i}" for i in range(1, 9)]
    EOL_PROBABILITY: float = 1e-6
    ZERO_HAZARD: float = 1e-6
    EOL_THRESHOLD: float = 0.8  # end of life threshold (80% of the intial capacity)
