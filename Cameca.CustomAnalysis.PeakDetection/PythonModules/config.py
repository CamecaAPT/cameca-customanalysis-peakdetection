import os
import tempfile
from contextlib import contextmanager

import yaml

DEFAULT_PREDICTION_ARGS = {
    "device": "cpu",
    "verbose": True,
    "freeze": None,
    "save": False,
    "conf": 0.05,
    "iou": 0.01,
    "max_det": 2000,
    "half": False,
    "single_cls": True,
}


@contextmanager
def temp_predition_config_file(data):
    prediction_args = {
        **DEFAULT_PREDICTION_ARGS,
        **data,
    }
    with temp_config_file(prediction_args) as f:
        yield f


@contextmanager
def temp_config_file(data):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
    try:
        with open(temp_file.name, "w") as f:
            yaml.dump(data, f)

        # Yield the path of the temporary file for further use
        yield temp_file.name
    finally:
        try:
            os.remove(temp_file.name)
        except OSError:
            pass  # Ignore any errors that occur during file deletion
