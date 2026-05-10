import logging

import numpy as np
import numpy.typing as npt
import torch

logger = logging.getLogger(__name__)


def map01(spectrum):
    return (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())


def build_histogram(
    mass_data: npt.NDArray[np.float32], bin_width: float, lower: float, bin_count: int
) -> npt.NDArray[np.int32]:
    return np.histogram(mass_data, bin_count, (lower, lower + (bin_width * bin_count)))[0]


def load_apt(spectrum: npt.NDArray[np.int32]):
    logger.debug("load_apt")
    logger.debug(spectrum)
    spectrum_log = np.log(spectrum + 1)
    spectrum_log = map01(spectrum_log)
    spectrum_log = torch.tensor(spectrum_log, dtype=torch.float32)
    return spectrum, spectrum_log
