import typing
import pyapsuite
import sys
import numpy as np
import torch
from pathlib import Path
from pyapsuite.pyapsuite_functions import fill_array_from_memory
import fast_histogram
from peak_detection.RangingNN.predictor import DetectionPredictor
import System
import Cameca.CustomAnalysis.Interface
import tempfile
import yaml
import os
from contextlib import contextmanager
import pkg_resources
import shutil

DEFAULT_PREDICTION_ARGS = {
    "device": "cpu",
    "verbose": True,
    "freeze": None,
    "save": False,
    "conf": 0.02,
    "iou": 0.01,
    "max_det": 2000,
    "half": False,
    "single_cls": True,
}

def main(context: pyapsuite.APSuiteContext, histogram: System.ReadOnlyMemory[System.Single]):
    spectrum = convert_histogram_data(histogram);
    spectrum_log = np.log(spectrum+1)
    spectrum_log = map01(spectrum_log)
    spectrum_log = torch.tensor(spectrum_log, dtype=torch.float32) 

    extension_dir = Path.cwd()
    save_dir = str(extension_dir)
    modelpath =  pkg_resources.resource_filename('peak_detection', 'RangingNN/modelweights/best.pt')
    
    prediction_args = {
        **DEFAULT_PREDICTION_ARGS,
        **{
            "conf": context.properties.Confidence,
            "iou": context.properties.IntersectionOverUnion,
            "max_det": context.properties.MaxDetections,
        },
    }
    with temp_config_file(prediction_args) as config_path:
        predictor = DetectionPredictor(modelpath, spectrum_log[None, None, ...], save_dir = save_dir, cfg = config_path)
        result = predictor()[0]
        peak_pred = result[:,:2].cpu()

        for p in peak_pred:
            p[p<0]=0

        return peak_pred

@contextmanager
def temp_config_file(data):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
    try:
        with open(temp_file.name, 'w') as f:
            yaml.dump(data, f)
        
        # Yield the path of the temporary file for further use
        yield temp_file.name
    finally:
        try:
            os.remove(temp_file.name)
        except OSError:
            pass  # Ignore any errors that occur during file deletion



def convert_histogram_data(data: System.ReadOnlyMemory[System.Single]) -> np.ndarray:
    count = data.Length
    np_array = np.empty(count, dtype=np.float32)
    fill_array_from_memory(np_array, data)
    return np_array

def map01(spectrum):
    return (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())


def fix_length(arr, length):
    if len(arr) < length:
        arr = np.pad(arr, (0, length - len(arr)), 'constant')
    elif len(arr) > length:
        arr = arr[:length]
    return arr