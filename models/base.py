from enum import Enum

class TrainMethod(Enum):
    SGD = "sgd"
    BATCH = "batch"
    CLOSED_FORM = "closed_form"