import importlib
import importlib.resources
import logging
import os
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from cameca_apt.extension import CancellationError, host
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


@dataclass
class TrainingData:
    ions: npt.NDArray[np.int32]
    counts: npt.NDArray[np.float64]
    mc: npt.NDArray[np.float64]
    encoder: dict[str, int]
    decoder: dict[int, str]


training_data: TrainingData | None = None


def ensure_training_data(progressToken: str | int | None = None) -> TrainingData:
    global training_data
    if training_data is None:
        logger.debug("Training data not initialized: Load files and build TrainingData object")
        # TODO: parameterize threshold? -- would need to introduce updatable state management of trained data on C# side
        # instead of one time initialization
        threshold_c = 1e-8

        # Load the ion training data once (this takes more time than training the model)
        data_resource = importlib.resources.files("peak_detection").joinpath(
            "IonClassifier", "training_data", "NewData", "Data0001"
        )
        with importlib.resources.as_file(data_resource) as training_data_path:
            data = load_training_data_from_csv_files(str(training_data_path), threshold_c, progressToken)
            training_data = TrainingData(
                data[0],
                data[1],
                data[2],
                data[3],
                data[4],
            )
    else:
        logger.debug("Training data already exists initialized: Returning existing object")
    return training_data


def load_full_training_data(
    progress,
) -> tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    dict[str, int],
    dict[int, str],
]:
    """Load training data for all elements

    Returns seperate arrays for each column to facilitate Python.NET serialization.
    AP Suite Python extensions are stateless, so each Python call states a clean instance.
    To maintain state across calls (to only have to load data once), pass the data back to the C# component
    of the extension to store and include in each Python call.

    Include encoder and decoder so the loaded data only has to include smaller encoded ints

    @Returns (encoded_ions, counts, mc, encoder, decoder)
    """
    # TODO: parameterize threshold? -- would need to introduce updatable state management of trained data on C# side instead of one time initialization
    threshold_c = 1e-8

    # Load the ion training data once (this takes more time than training the model)
    data_resource = importlib.resources.files("peak_detection").joinpath(
        "IonClassifier", "training_data", "NewData", "Data0001"
    )
    with importlib.resources.as_file(data_resource) as training_data_path:
        return load_training_data_from_csv_files(str(training_data_path), threshold_c, progress)


def load_training_data_from_csv_files(path, threshold_c, progressToken: str | int | None = None):
    df_all = []
    norm_columns = ["counts"]
    files = os.listdir(path)
    count = len(files)
    log_split = count / 10
    for i, file in enumerate(files):
        df = pd.read_csv(os.path.join(path, file), keep_default_na=False)
        # drop unused columns to reduce needed space
        df.drop(columns=["charge", "ion2", "charge2"], inplace=True)
        # Min-max normalization of counts
        df[norm_columns] = MinMaxScaler().fit_transform(df[norm_columns])
        # ignoring peak with very low prominence
        df.drop(df[df["counts"] <= threshold_c].index, inplace=True)

        df_all.append(df)

        if i % log_split == 0:
            logger.debug("Loading files %d / %d", i, count)

        if progressToken is not None:
            host.report_progress(progressToken, float(i) / count)

        if host.is_cancelled:
            raise CancellationError("Loading training data cancelled by user")
    # Report finished
    if progressToken is not None:
        host.report_progress(progressToken, 1.0)

    # Create single dataframe for later filtering
    df = pd.concat(df_all, ignore_index=True)

    # encoding
    encoder, decoder = make_RF_encoder(df["ion"].unique())

    encoded_ions = df["ion"].apply(encoder.get)
    return (
        encoded_ions.to_numpy(),
        df["counts"].to_numpy(),
        df["mc"].to_numpy(),
        encoder,
        decoder,
    )


def make_RF_encoder(unique_ions):
    target_encoder = dict()
    target_decoder = dict()
    for i, val in enumerate(unique_ions):
        target_encoder[val] = i
        target_decoder[i] = val
    return target_encoder, target_decoder
