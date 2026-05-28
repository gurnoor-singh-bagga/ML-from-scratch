from enum import Enum

class FillStrategy(Enum):
    DROP="drop"
    MEAN="mean"
    MEDIAN="median"
    MEDIAN_STOCHASTIC="median_stochastic"
    