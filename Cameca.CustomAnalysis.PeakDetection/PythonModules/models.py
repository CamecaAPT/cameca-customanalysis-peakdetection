from dataclasses import dataclass


@dataclass
class EntryReturn:
    peak_pred: list[list[float]]
    elem1: list[str]
    conf1: list[float]
    elem2: list[str]
    conf2: list[float]
    peak_iter: list[int]
