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
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.neighbors import KernelDensity
import torch.nn.functional as F
from enum import Enum
import joblib
import configparser
import re
from collections import Counter
import pandas as pd
import importlib.resources
import json
from typing import Tuple

# Match for splitting ion formulas into component elements: matches right before any uppercase character
# Lookbehind requires one character before, blocking matching at the beginning of the text, causing the
# split to not consider the beginning a match and split which will otherwise include an initial empty string
pattern = re.compile(r'(?<=.)(?=[A-Z])')

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

def load_full_training_data(progress) -> Tuple[np.ndarray[np.int32], np.ndarray[np.float64], np.ndarray[np.float64], dict[str, int], dict[int, str]]:
    """Load training data for all elements
    
    Returns seperate arrays for each column to facilitate Python.NET serialization.
    AP Suite Python extensions are stateless, so each Python call states a clean instance.
    To maintain state across calls (to only have to load data once), pass the data back to the C# component
    of the extension to store and include in each Python call.

    Include encoder and decoder so the loaded data only has to include smaller encoded ints

    @Returns (encoded_ions, counts, mc, encoder, decoder)
    """
    # TODO: parameterize threshold? -- would need to introduce updatable state management of trained data on C# side instead of one time initialization
    threshold_c=1e-8
    
    # Load the ion training data once (this takes more time than training the model)
    data_resource = importlib.resources.files('peak_detection').joinpath('IonClassifier', 'training_data', 'NewData', 'Data0001')
    with importlib.resources.as_file(data_resource) as training_data_path:
        return load_training_data_from_csv_files(str(training_data_path), threshold_c, progress)


def load_training_data_from_csv_files(path, threshold_c, progress):
    df_all = []
    norm_columns = ["counts"]
    files = os.listdir(path)
    count = len(files)
    for i, file in enumerate(files):
        df = pd.read_csv(os.path.join(path, file), keep_default_na=False)
        # drop unused columns to reduce needed space
        df.drop(columns=["charge", "ion2", "charge2"], inplace=True)
        # Min-max normalization of counts
        df[norm_columns] = MinMaxScaler().fit_transform(df[norm_columns])
        # ignoring peak with very low prominence
        df.drop(df[df["counts"] <= threshold_c].index, inplace=True)

        df_all.append(df)
        progress.Report(float(i) / count)

    # Create single dataframe for later filtering
    df = pd.concat(df_all, ignore_index=True)

    # encoding
    encoder, decoder = make_RF_encoder(df["ion"].unique())

    encoded_ions = df["ion"].apply(encoder.get)
    return (encoded_ions.to_numpy(), df["counts"].to_numpy(), df["mc"].to_numpy(), encoder, decoder)

def to_dict(clr_dict):
    res = {}
    for pair in clr_dict:
        res[pair.Key] = pair.Value
    return res

def recommend_elements(ions,
                       counts,
                       mc,
                       encoder,
                       decoder,
                       peak_range_pred,
                       elements,
                       thresh,
                       num_elements,
                       logger):
    # Convert CLR objects to Python
    full_ions = to_ndarray(ions, dtype=np.int32);
    full_ion_counts = to_ndarray(counts, dtype=np.float64);
    full_mc = to_ndarray(mc, dtype=np.float64);
    encoder = to_dict(encoder)
    decoder = to_dict(decoder)
    encoded_elem = [encoder[x] for x in elements]

    pred_arr = to_ndarray(peak_range_pred, np.float64).reshape((-1, 2)) / 0.01
    
    element_only = filter_to_elements(encoder.keys())
    logger(",".join(set(element_only)))
    logger(",".join(set(elements)))
    merged = list(set(elements).union(set(element_only)))
    logger(",".join(merged))
    df = create_filtered_dataframe(logger, full_mc, full_ion_counts, full_ions, encoder, merged)
    logger(str(df))
    logger(", ".join(str(x) for x in df["target"].unique()))
    res = suggest_additional_elements(df, pred_arr, encoded_elem, thresh, num_elements, logger)
    return list(decoder[x] for x in res)


def main(context: pyapsuite.APSuiteContext,
         histogram: System.ReadOnlyMemory[System.Double],
         element_list,
         elements_to_get_molecules,
         confidence,
         intersectionOverUnion,
         maxDetections,
         iterations,
         ions,
         counts,
         mc,
         encoder,
         decoder,
         spectrum_log,
         logger):
    # Convert CLR objects to Python
    full_ions = to_ndarray(ions, dtype=np.int32);
    full_ion_counts = to_ndarray(counts, dtype=np.float64);
    full_mc = to_ndarray(mc, dtype=np.float64);
    encoder = to_dict(encoder)
    decoder = to_dict(decoder)

    spectrum, spectrum_log = load_apt(histogram)

    extension_dir = Path.cwd()
    save_dir = str(extension_dir)

    modelpath =  pkg_resources.resource_filename('peak_detection', 'RangingNN/modelweights/best_v0_2025-11-12.pt')
    
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

    # Multiple iterations
    peak_range_toadd = list()
    counts_toadd = list()
    for i in range(iterations - 1):
        logger(f"Running iteration {i + 1}")
        multiplier = 0.01

        # Processing spectrum part 1
        n = spectrum_log.shape[0]
        x1 = np.arange(n) * multiplier   # equivalent to your linspace
        # peak_range_pred: list of (start, end)
        ranges = np.asarray(peak_pred, dtype=float)   # shape (R, 2)
        starts = ranges[:, 0] * multiplier                  # shape (R,)
        ends   = ranges[:, 1] * multiplier                  # shape (R,)

        # Broadcasting: shape (n, R), then reduce along R
        in_any_range = np.logical_or.reduce(
            (x1[:, None] > starts[None, :]) &
            (x1[:, None] < ends[None, :]),
            axis=1
        )

        idx_delete = x1[in_any_range]

        # Processing spectrum part 2
        x1 = np.asarray(x1)
        spectrum_log = np.asarray(spectrum_log)
        mask = np.isin(x1, idx_delete)

        spectrum_log_mod = spectrum_log.copy()
        spectrum_log_mod[mask] = 0.2

        spectrum_log_mod = torch.Tensor(spectrum_log_mod)

        # Running predictions
        with temp_config_file(prediction_args) as config_path:
            predictor_mod = DetectionPredictor(modelpath, spectrum_log_mod[None, None, ...], save_dir = save_dir, cfg = config_path)
            result_mod = predictor_mod()[0]
            peak_range_pred_mod = result_mod[:,:2].cpu()
            counts_mod = result_mod[:,2].cpu()

        # Collect any new peaks and append to the original list
        tol = 0.5
        for i, count in zip(peak_range_pred_mod, counts_mod):
            start = float(i[0])
            end = float(i[1])
            max_iou = 1.0
            min_dist = 1000
            for j in peak_pred.tolist():

                start2 = float(j[0])
                end2 = float(j[1])
                iou = calculate_iou_1d(interval1=[start, end], interval2=[start2, end2])
                if iou < max_iou:
                    max_iou = iou

                if multiplier*abs(float(start)-float(start2)) < min_dist:
                    min_dist = multiplier*abs(float(start)-float(start2))
            # Want max iou to be 0 -> totally new peak
            if max_iou == 0.0:
                if min_dist > tol:
                    peak_range_toadd.append([start, end])
                    counts_toadd.append(count)

        # Update existing preditions and counts
        peak_pred = peak_pred.tolist()
        counts = counts.tolist()

        for i, j in zip(peak_range_toadd, counts_toadd):
            peak_pred.append(i)
            counts.append(j)

        peak_pred = torch.Tensor(peak_pred)
        counts = torch.Tensor(counts)

    # Apply element filtering to cached full element data
    df = create_filtered_dataframe(logger, full_mc, full_ion_counts, full_ions, encoder, element_list, elements_to_get_molecules)

    # Train the model
    scaler, model = train_RF_model(df, n_estimators=25, random_state=42)
    logger(str(model))

    elem1, elem2, conf1, conf2, _ = run_RF_model(peak_pred,
                                                    counts,
                                                    scaler,
                                                    model,
                                                    decoder,
                                                    logger,
                                                    use_peak_maxima=True,
                                                    spectrum_log=spectrum_log)

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

def to_ndarray(data, dtype) -> np.ndarray:
    count = data.Length
    np_array = np.empty(count, dtype=dtype)
    fill_array_from_memory(np_array, data)
    return np_array

def map01(spectrum):
    return (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())

def load_apt(histogram: System.ReadOnlyMemory[System.Double]):
    spectrum = to_ndarray(histogram, dtype=np.float64);
    spectrum_log = np.log(spectrum+1)
    spectrum_log = map01(spectrum_log)
    spectrum_log = torch.tensor(spectrum_log, dtype=torch.float32)
    return spectrum, spectrum_log

def run_RF_model(peak_range_pred,
                 counts,
                 scaler,
                 model,
                 decoder,
                 logger,
                 use_peak_maxima=False,
                 spectrum_log=None):

    multiplier = 0.01

    if not use_peak_maxima:
        X_0 = multiplier*peak_range_pred[:, 0]
    else:
        # Need to use the predicted peak ranges and data file to get the m/c position of the peak maximum
        X_0 = list()
        for i in range(peak_range_pred.shape[0]):
            start = multiplier*peak_range_pred[i, 0]
            end = multiplier*peak_range_pred[i, 1]
            peak_max = 0
            max_ind = 0
            for count, j in enumerate([(multiplier * i) + (multiplier / 2) for i in range(0, int(307.2 / multiplier))]):
                if j >= start and j <= end:
                    if spectrum_log[count] > peak_max:
                        peak_max = spectrum_log[count]
                        max_ind = j
            #print('Have peak max', peak_max, 'at max ind', max_ind, 'for peak range', start, end)
            X_0.append(max_ind)
    X_1 = np.array(counts)

    X = np.column_stack((X_0, X_1))

    X_norm = scaler.transform(X)

    preds = model.predict_proba(X_norm)

    elements = list()
    second_elements = list()
    confs = list()
    second_confs = list()
    for pred in preds:
        pred_index1 = pred.argsort()[-1]
        element = decoder[model.classes_[pred_index1]]
        conf = pred[pred_index1]
        elements.append(element)
        confs.append(conf)
        
        pred_index2 = pred.argsort()[-2]
        second_element = decoder[model.classes_[pred_index2]]
        second_conf = pred[pred_index2]
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

def make_RF_encoder(unique_ions):
    target_encoder = dict()
    target_decoder = dict()
    for i, val in enumerate(unique_ions):
        target_encoder[val] = i
        target_decoder[i] = val
    return target_encoder, target_decoder


def train_RF_model(df, n_estimators=25, random_state=42):
    features = [col for col in df.columns if col not in ['ions', 'target']]

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    y = df['target']
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)
    return scaler, model

def filtered_encoded(elem, mol, encoder, logger):
    elem_set = set(elem)
    mol_set = set(mol)
    results = []
    for ion, encoded in encoder.items():
        comp_elem = pattern.split(ion)
        all_comp_in_elem = all(x in elem_set for x in comp_elem)
        mol_in_comp = any(x in mol_set for x in pattern.split(ion))
        if (ion in elem_set) or (all_comp_in_elem and mol_in_comp):
            results.append(encoded)
    return results

def filter_to_elements(names):
    """Filters a list of ion names to only elements by removing all complex ions"""
    elements = []
    for ion in names:
        if not pattern.search(ion):
            elements.append(ion)
    return elements

def create_filtered_dataframe(logger, mc, counts, ions, encoder, element_list, elements_to_get_molecules=None):
    elements_to_get_molecules = [] if elements_to_get_molecules is None else elements_to_get_molecules
    matches = filtered_encoded(element_list, elements_to_get_molecules, encoder, logger)
    mask = np.isin(ions, matches)
    return pd.DataFrame({
        "mc": mc[mask],
        "counts": counts[mask],
        "target": ions[mask],
    })

def create_kde(series):
    return KernelDensity(kernel='gaussian', bandwidth=0.5).fit(series.array.reshape(-1, 1))

def make_lookup_model(element_df):
    # Group by "target" column (the encoded elements), then for each grouped series of mc values,
    # create a KernalDensity instance to return
    # Finally map as a dictionary for expected output shape
    return element_df.groupby("target")["mc"].apply(create_kde).to_dict()

def suggest_additional_elements(elements_df, peak_range_pred, elements, thresh, num_elements, logger):
    """Returns set of encoded element ions that are good candidates for adding to the element list"""
    # Lookup model is then created from full training data of all basic elements
    lookup_model = make_lookup_model(elements_df)

    full_adds = set()
    for pred, element in zip(peak_range_pred, elements):
        conf = np.exp(lookup_model[element].score_samples(pred.reshape(-1,1)/100))[0]

        logger(f"{pred=}, {element=}, {conf=}, {thresh=}")

        if conf < thresh:
            add_ions = lookup_add_ions(lookup_model=lookup_model,
                                            x=pred[0].reshape(-1,1)/100,
                                            thresh=thresh,
                                            logger=logger)
            logger(str(add_ions))
            logger(",".join([str(x) for x in add_ions]))
            for a in add_ions[:num_elements]:
                full_adds.add(a)
    logger(",".join([str(x) for x in full_adds]))
    return full_adds

def lookup_add_ions(lookup_model, x, thresh, logger):
    ranking_dict = dict()
    for k in lookup_model.keys():
        if k not in ranking_dict.keys():
            log_prob = lookup_model[k].score_samples(x)     # log(f(x0))
            pdf_value = np.exp(log_prob)[0]
            ranking_dict[k] = pdf_value
            
    prediction_rankings = sorted(ranking_dict.items(), key=lambda x: x[1], reverse=True)
    logger(f"{prediction_rankings=}")
    return [i for i, c in prediction_rankings if c > thresh]

def calculate_iou_1d(interval1, interval2):
    """
    Calculates the Intersection over Union (IoU) of two 1D intervals.

    Args:
        interval1: A tuple or list of two numbers representing the start and end of the first interval.
        interval2: A tuple or list of two numbers representing the start and end of the second interval.

    Returns:
        The IoU of the two intervals, a float between 0 and 1.
    """

    start1, end1 = interval1
    start2, end2 = interval2

    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)

    intersection_length = max(0, intersection_end - intersection_start)

    union_length = (end1 - start1) + (end2 - start2) - intersection_length

    if union_length == 0:
        return 0.0

    iou = intersection_length / union_length
    return iou