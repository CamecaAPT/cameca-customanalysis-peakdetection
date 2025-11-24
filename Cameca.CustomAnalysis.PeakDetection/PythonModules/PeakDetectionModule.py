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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from enum import Enum
import joblib
import configparser
import re
from collections import Counter
import pandas as pd
import importlib.resources
import json


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

def main(context: pyapsuite.APSuiteContext,
         histogram: System.ReadOnlyMemory[System.Double],
         element_list,
         elements_to_get_molecules,
         confidence,
         intersectionOverUnion,
         maxDetections):
    spectrum, spectrum_log = load_apt(histogram)

    extension_dir = Path.cwd()
    save_dir = str(extension_dir)

    modelpath =  pkg_resources.resource_filename('peak_detection', 'RangingNN/modelweights/best.pt')
    
    prediction_args = {
        **DEFAULT_PREDICTION_ARGS,
        **{
            "conf": confidence,
            "iou": intersectionOverUnion,
            "max_det": maxDetections,
        },
    }
    with temp_config_file(prediction_args) as config_path:
        predictor = DetectionPredictor(modelpath, spectrum_log[None, None, ...], save_dir = save_dir, cfg = config_path)
        result = predictor()[0]
        peak_pred = result[:,:2].cpu()
        counts = result[:,2].cpu()

    # Load the ion training data once (this takes more time than training the model)
    data_resource = importlib.resources.files('peak_detection').joinpath('IonClassifier', 'training_data', 'NewData', 'Data0001')
    with importlib.resources.as_file(data_resource) as training_data_path:
        mc, ion_counts, ions = load_ion_training_data(path=str(training_data_path),
                                                    element_list = element_list,
                                                    elements_to_get_molecules=elements_to_get_molecules,
                                                    threshold_c= 1e-8,
                                                    num_files=10000)
    # Train the model
    scaler, model, target_decoder = create_RF_model(mc=mc,
                                                    counts=ion_counts,
                                                    ions=ions,
                                                    n_estimators=25,
                                                    random_state=42) # Change to False in subsequent runs
    elem1, elem2, conf1, conf2, _ = run_RF_model(peak_pred,
                                                    counts,
                                                    scaler,
                                                    model,
                                                    target_decoder)
    return peak_pred, elem1, conf1, elem2, conf2


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



def convert_histogram_data(data: System.ReadOnlyMemory[System.Double]) -> np.ndarray:
    count = data.Length
    np_array = np.empty(count, dtype=np.float64)
    fill_array_from_memory(np_array, data)
    return np_array

def map01(spectrum):
    return (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())

def load_apt(histogram: System.ReadOnlyMemory[System.Double]):
    spectrum = convert_histogram_data(histogram);
    spectrum_log = np.log(spectrum+1)
    spectrum_log = map01(spectrum_log)
    spectrum_log = torch.tensor(spectrum_log, dtype=torch.float32)
    return spectrum, spectrum_log

def run_RF_model(peak_range_pred,
                 counts,
                 scaler,
                 model,
                 target_decoder):

    multiplier = 0.01

    X_0 = multiplier*peak_range_pred[:, 0]
    X_1 = np.array(counts)

    X = np.column_stack((X_0, X_1))

    X_norm = scaler.transform(X)

    preds = model.predict_proba(X_norm)

    elements = list()
    second_elements = list()
    confs = list()
    second_confs = list()
    for pred in preds:
        element = target_decoder[pred.argsort()[-1]]
        conf = pred[pred.argsort()[-1]]
        elements.append(element)
        confs.append(conf)

        second_element = target_decoder[pred.argsort()[-2]]
        second_conf = pred[pred.argsort()[-2]]
        if second_conf > 0.0:
          second_elements.append(second_element)
        else:
          second_elements.append('NaN')
        second_confs.append(second_conf)

    return elements, second_elements, confs, second_confs, X

def get_model_data(model_data_path, model_data_key):
    """ (reduced_set, include_chargestates, include_molecules, peak_ratio_neighbor_width, max_peaks) """
    config = configparser.ConfigParser()
    config.read(model_data_path)
    reduced_set = config.getboolean(model_data_key, 'reduced_set')
    include_chargestates = config.getboolean(model_data_key, 'include_chargestates')
    include_molecules = config.getboolean(model_data_key, 'include_molecules')
    peak_ratio_neighbor_width = config.getint(model_data_key, 'peak_ratio_neighbor_width', fallback=None)
    max_peaks = config.getint(model_data_key, 'max_peaks', fallback=None)
    return (reduced_set, include_chargestates, include_molecules, peak_ratio_neighbor_width, max_peaks)


def load_ion_training_data(path,
                           element_list,
                           elements_to_get_molecules,
                           threshold_c=1e-8,
                           num_files=10000):
    """
    load the evaluation files, get input and gt, normalized counts,
    """

    mc_all = list()
    counts_all = list()
    ions_all = list()

    files = os.listdir(path)[0:num_files]
    for file in files:
        df = pd.read_csv(os.path.join(path, file), keep_default_na=False)

        mc = df.get(['mc']).to_numpy().squeeze()
        counts = df.get(['counts']).to_numpy().squeeze()
        target = {'ion': df.get(['ion']).to_numpy().squeeze(), 'charge': df.get(['charge']).to_numpy().squeeze(),
                  'ion2': df.get(['ion2']).to_numpy().squeeze(),
                  'charge2': df.get(['charge2']).to_numpy().squeeze()}
        counts = ( counts - counts.min() ) / (counts.max() - counts.min()) # normalize counts from one spectrum to 0-1
        labels = np.array([re.findall('.[^A-Z]*', str(it))[0] for it in target['ion']]) # Not including the light elements in mole here

        indexes = counts > threshold_c # ignoring peak with very low prominence

        ions = target['ion']
        charges = target['charge']

        mc = mc[indexes]
        counts = counts[indexes]
        ions = ions[indexes]
        charges = charges[indexes]

        inds_keep = list()
        if element_list != 'all':
            for i, ion in enumerate(ions):
                if ion in element_list:
                    inds_keep.append(i)

        if element_list == 'all':
            inds_keep = list(range(len(ions)))

        for ion in elements_to_get_molecules:
            for i, ion2 in enumerate(ions):
                if ion in ion2 and ion != ion2:
                    if i not in inds_keep:
                        inds_keep.append(i)

        mc = mc[inds_keep]
        counts = counts[inds_keep]
        ions = ions[inds_keep]
        charges = charges[inds_keep]

        mc_all.append(mc)
        counts_all.append(counts)
        ions_all.append(ions)

    all_mc_flat = np.concatenate(mc_all)
    all_counts_flat = np.concatenate(counts_all)
    all_ions_flat = np.concatenate(ions_all)

    return all_mc_flat, all_counts_flat, all_ions_flat


def create_RF_model(mc,
                    counts,
                    ions,
                    threshold_c=1e-8,
                    n_estimators=25,
                    random_state=42):
    target_encoder, target_decoder = make_RF_encoder(unique_ions=np.unique(ions))

    scaler, model = train_RF_model(target_encoder=target_encoder,
                                   mc=mc,
                                   counts=counts,
                                   ions=ions,
                                   n_estimators=n_estimators,
                                   random_state=random_state)

    return scaler, model, target_decoder


def make_RF_encoder(unique_ions):
    print('Getting RF encoder')
    target_encoder = dict()
    target_decoder = dict()
    for i, val in enumerate(unique_ions):
        target_encoder[val] = i
        target_decoder[i] = val
    return target_encoder, target_decoder


def train_RF_model(target_encoder, mc, counts, ions, n_estimators=25, random_state=42):
    print('Training RF model')
    df = pd.DataFrame({'mc': mc,
                   'counts': counts,
                   'ions': ions,
                   'target': [target_encoder[i] for i in ions]})
    features = [col for col in df.columns if col not in ['ions', 'target']]

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    y = df['target']
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)
    return scaler, model