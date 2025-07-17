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
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from enum import Enum
import joblib
import configparser
import re
from collections import Counter

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

class PeakDetectionImplementation(Enum):
    NeuralNetwork = 1
    RandomForest = 2
    RandomForestReduced = 3


def main(context: pyapsuite.APSuiteContext, histogram: System.ReadOnlyMemory[System.Double]):
    spectrum, spectrum_log = load_apt(histogram)

    extension_dir = Path.cwd()
    save_dir = str(extension_dir)

    implementation = PeakDetectionImplementation(context.properties.Implementation)

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
        counts = result[:,2].cpu()

    model_path = os.path.expandvars(context.properties.ModelPath)
    # Check if the model is set and exists
    if not os.path.isfile(model_path):
        raise RuntimeError("ModelPath not set or file does not exist: %s" % model_path)
    
    if implementation == PeakDetectionImplementation.NeuralNetwork:
        res, confidence, profile_final = predict_peak_ions(model_path, spectrum, peak_pred, bin_width=0.01, max_width_Da=0.5)
        return peak_pred, res, confidence, None, None
    elif implementation == PeakDetectionImplementation.RandomForest:
        # RandomForest requires additional scalar model file
        scalar_path = os.path.expandvars(context.properties.ScalarPath)
        if not os.path.isfile(scalar_path):
            raise RuntimeError("ScalarPath not set or file does not exist: %s" % scalar_path)
        model_data_path = os.path.join(extension_dir, 'ModelData.ini')
        elem1, elem2, conf1, conf2 = random_forest(peak_pred, counts, scalar_path, model_path, model_data_path)
        return peak_pred, elem1, conf1, elem2, conf2

    raise RuntimeError("Unsupported implementation: %d" % implementation)


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


def fix_length(arr, length):
    if len(arr) < length:
        arr = np.pad(arr, (0, length - len(arr)), 'constant')
    elif len(arr) > length:
        arr = arr[:length]
    return arr

def load_apt(histogram: System.ReadOnlyMemory[System.Double]):
    spectrum = convert_histogram_data(histogram);
    spectrum_log = np.log(spectrum+1)
    spectrum_log = map01(spectrum_log)
    spectrum_log = torch.tensor(spectrum_log, dtype=torch.float32)
    return spectrum, spectrum_log


def get_label_encoder():
    CHEMICAL_ELEMENTS = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
                         'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
                         'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
                         'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                         'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd',
                         'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
                         'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'U']

    le = LabelEncoder()
    le.fit(CHEMICAL_ELEMENTS)
    return le
    

def get_RF_encoder(reduced_set=False, include_chargestates=False, include_molecules=False):
    if include_chargestates == False:
        if reduced_set == False:
            if include_molecules == False:
                target_decoder = {0: 'Ag', 1: 'Al', 2: 'Ar', 3: 'As', 4: 'Au', 5: 'B', 6: 'Ba',
                              7: 'Be', 8: 'Bi', 9: 'Br', 10: 'C', 11: 'Ca', 12: 'Cd', 13: 'Ce',
                              14: 'Cl', 15: 'Co', 16: 'Cr', 17: 'Cs', 18: 'Cu', 19: 'Dy', 20: 'Er',
                              21: 'Eu', 22: 'F', 23: 'Fe', 24: 'Ga', 25: 'Gd', 26: 'Ge', 27: 'H',
                              28: 'He', 29: 'Hf', 30: 'Hg', 31: 'Ho', 32: 'I', 33: 'In', 34: 'Ir',
                              35: 'K', 36: 'Kr', 37: 'La', 38: 'Li', 39: 'Lu', 40: 'Mg', 41: 'Mn',
                              42: 'Mo', 43: 'N', 44: 'Na', 45: 'Nb', 46: 'Nd', 47: 'Ni', 48: 'O',
                              49: 'Os', 50: 'P', 51: 'Pb', 52: 'Pd', 53: 'Pr', 54: 'Pt', 55: 'Rb',
                              56: 'Re', 57: 'Rh', 58: 'Ru', 59: 'S', 60: 'Sb', 61: 'Sc', 62: 'Se',
                              63: 'Si', 64: 'Sm', 65: 'Sn', 66: 'Sr', 67: 'Ta', 68: 'Tb', 69: 'Te',
                              70: 'Th', 71: 'Ti', 72: 'Tl', 73: 'Tm', 74: 'U', 75: 'V', 76: 'W',
                              77: 'Xe', 78: 'Y', 79: 'Zn', 80: 'Zr'}
            else:
                raise ValueError('Not supported yet')
        else:
            if include_molecules == False:
                target_decoder = {0: 'Ag', 1: 'Al', 2: 'Au', 3: 'C', 4: 'Ca', 5: 'Co', 6: 'Cu',
                              7: 'Fe', 8: 'Ga', 9: 'H', 10: 'Li', 11: 'Mg', 12: 'Mn', 13: 'Na',
                              14: 'Ni', 15: 'O', 16: 'Pd', 17: 'Pt', 18: 'S', 19: 'Sc', 20: 'Si',
                              21: 'Ti', 22: 'Zn', 23: 'Zr'}
            else:
                # The newest target decoder includes HZr, but previously trained reducedset_molecules model missed this one
                target_decoder = {0: 'Ag', 1: 'Al', 2: 'AlAlO', 3: 'Au', 4: 'AuHH',
                                  5: 'AuHHHHO', 6: 'AuN', 7: 'AuNN', 8: 'AuZn', 9: 'C', 10: 'CC', 11: 'CCC',
                                  12: 'CN', 13: 'Ca', 14: 'Co', 15: 'CoO', 16: 'CoOO', 17: 'CoOOO', 18: 'Cu',
                                  19: 'Fe', 20: 'Ga', 21: 'H', 22: 'HH', 23: 'HHH', 24: 'HHO', 25: 'HO',
                                  26: 'HOO', 27: 'Li', 28: 'Mg', 29: 'Mn', 30: 'MnO', 31: 'MnOO', 32: 'MnOOH',
                                  33: 'MnOOO', 34: 'MnOOOH', 35: 'MnOOOO', 36: 'MnOOOOH', 37: 'NN', 38: 'Na',
                                  39: 'Ni', 40: 'NiNiO', 41: 'NiO', 42: 'NiOO', 43: 'NiOOO', 44: 'O',
                                  45: 'OO', 46: 'Pd', 47: 'Pt', 48: 'S', 49: 'Sc', 50: 'Si', 51: 'Ti', 52: 'Zn', 53: 'Zr'}
    else:
        if reduced_set == False:
            raise ValueError('Not supported yet')
        else:
            if include_molecules == False:
                target_decoder = {0: 'Ag1+',1: 'Al1+', 2: 'Al2+', 3: 'Al3+', 4: 'Au1+', 5: 'Au2+', 6: 'C1+',
                              7: 'C2+', 8: 'Ca1+', 9: 'Ca2+', 10: 'Co1+', 11: 'Co2+', 12: 'Cu1+',
                              13: 'Cu2+', 14: 'Fe1+', 15: 'Fe2+', 16: 'Fe3+', 17: 'Ga1+', 18: 'Ga2+',
                              19: 'H1+', 20: 'Li1+', 21: 'Li2+', 22: 'Mg1+', 23: 'Mg2+', 24: 'Mn1+',
                              25: 'Mn2+', 26: 'Na1+', 27: 'Ni1+', 28: 'Ni2+', 29: 'O1+', 30: 'O2+',
                              31: 'Pd1+', 32: 'Pd2+', 33: 'Pt1+', 34: 'Pt2+', 35: 'S1+', 36: 'S2+',
                              37: 'Sc1+', 38: 'Sc2+', 39: 'Si1+', 40: 'Si2+', 41: 'Ti1+', 42: 'Ti2+',
                              43: 'Zn1+', 44: 'Zn2+', 45: 'Zr1+', 46: 'Zr2+'}
            else:
                target_decoder = {0: 'Ag1+', 1: 'Al1+', 2: 'Al2+', 3: 'Al3+',
                                  4: 'AlAlO1+', 5: 'Au1+', 6: 'Au2+', 7: 'AuHH1+', 8: 'AuHHHHO1+', 9: 'AuN1+',
                                  10: 'AuNN1+', 11: 'AuZn1+', 12: 'C1+', 13: 'C2+', 14: 'CC1+', 15: 'CCC1+',
                                  16: 'CN1+', 17: 'Ca1+', 18: 'Ca2+', 19: 'Co1+', 20: 'Co2+', 21: 'CoO1+',
                                  22: 'CoOO1+', 23: 'CoOOO1+', 24: 'Cu1+', 25: 'Cu2+', 26: 'Fe1+', 27: 'Fe2+',
                                  28: 'Fe3+', 29: 'Ga1+', 30: 'Ga2+', 31: 'H1+', 32: 'HH1+', 33: 'HHH1+',
                                  34: 'HHO1+', 35: 'HO1+', 36: 'HOO1+', 37: 'HZr1+', 38: 'Li1+', 39: 'Li2+',
                                  40: 'Mg1+', 41: 'Mg2+', 42: 'Mn1+', 43: 'Mn2+', 44: 'MnO1+', 45: 'MnOO1+',
                                  46: 'MnOOH1+', 47: 'MnOOO1+', 48: 'MnOOOH1+', 49: 'MnOOOO1+', 50: 'MnOOOOH1+',
                                  51: 'NN1+', 52: 'Na1+', 53: 'Ni1+', 54: 'Ni2+', 55: 'NiNiO1+', 56: 'NiO1+',
                                  57: 'NiOO1+', 58: 'NiOOO1+', 59: 'O1+', 60: 'O2+', 61: 'OO1+', 62: 'Pd1+',
                                  63: 'Pd2+', 64: 'Pt1+', 65: 'Pt2+', 66: 'S1+', 67: 'S2+', 68: 'Sc1+', 69: 'Sc2+',
                                  70: 'Si1+', 71: 'Si2+', 72: 'Ti1+', 73: 'Ti2+', 74: 'Zn1+', 75: 'Zn2+', 76: 'Zr1+',
                                  77: 'Zr2+', 78: 'Zr3+'}
    return target_decoder

def normalize_encoder(encoder: dict[int, str]) -> dict[int, str]:
    return { x: normalize_formula(y) for x, y in encoder.items() }

def normalize_formula(formula: str) -> str:
    """Adjust formula strings ot make parsable by AP Suite"""
    no_charge_state = re.sub(r'\d+\+$', '', formula)
    matches = re.findall(r'[A-Z][a-z]*', no_charge_state)
    counter = Counter(matches)
    return ''.join([x if y <= 1 else x + str(y) for x, y in counter.items()])


def predict_elements(model, spectrum, label_encoder, device):
    model.eval()
    with torch.no_grad():
        spectrum = torch.FloatTensor(spectrum).unsqueeze(0).to(device)
        outputs = model(spectrum, lengths = [spectrum.shape[1]])

        probabilities = F.softmax(outputs, dim=2)
        predictions = torch.argmax(outputs, dim=2)

        # Convert numerical predictions to element names
        element_predictions = label_encoder.inverse_transform(predictions.cpu().numpy().ravel())
        confidence_scores = torch.max(probabilities, dim=2)[0].cpu().numpy().ravel()

        return element_predictions, confidence_scores
    

def predict_peak_ions(model_path, spectrum, peak_range_pred, bin_width=0.01, max_width_Da=0.5):
    '''
    Certain wide ranges due to the tail leads to over-high indensity,
    shrinking the other intensities, so set limit here
    '''
    profile = np.zeros_like(peak_range_pred)
    peak_range_pred = np.asarray(peak_range_pred)

    # Preprocess detected peak ranging data for IonClassifier model
    for k in range(profile.shape[0]):
        profile[k][0] = spectrum[round(peak_range_pred[k][0]):round(peak_range_pred[k][1])+1].argmax()
        profile[k][0] = (profile[k][0] + round(peak_range_pred[k][0]))/100

        window = min(max_width_Da/bin_width, round(peak_range_pred[k][1])+1-round(peak_range_pred[k][0]))
        profile[k][1] = spectrum[round(peak_range_pred[k][0]):round(peak_range_pred[k][0]+window)].sum()

    profile[:,1] = (profile[:,1] - profile[:,1].min() ) / (profile[:,1].max() - profile[:,1].min()) # normalize counts from one spectrum to 0-1

    # Make sure the profile is in sequential order, increasing m/c
    profile_final = profile[profile[:, 0].argsort()]

    # Run the IonClassifier model to predict the peak IDs
    RNNmodel = torch.load(model_path, map_location='cpu')['ema']
    le = get_label_encoder()
    res, confidence = predict_elements(RNNmodel, profile_final, le, 'cpu')

    return res, confidence, profile_final


def random_forest(peak_range_pred, counts, scaler_path, model_path, model_data_path):
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    
    model_key = os.path.splitext(os.path.basename(model_path))[0]
    (reduced_set, include_chargestates, include_molecules, peak_ratio_neighbor_width, max_peaks) = get_model_data(model_data_path, model_key)

    X_0 = 0.01*peak_range_pred[:, 0]
    X_1 = counts
    if peak_ratio_neighbor_width is not None and max_peaks is not None:
        ratios = list()
        mc_ratios = list()
        for i, count1 in enumerate(X_1):
            ratio_row = list()
            mc_ratio_row = list()
            for j, count2 in enumerate(X_1):
                #if j > i:
                ratio = count1 / count2
                mc_ratio = X_0[i] / X_0[j]
                dist = abs(X_0[j] - X_0[i])
                if dist <= peak_ratio_neighbor_width and ratio != 1.0:
                    ratio_row.append(ratio)
                    mc_ratio_row.append(mc_ratio)
            ratios.append(ratio_row)
            mc_ratios.append(mc_ratio_row)
        # Pad the ratio rows
        for i, lst in enumerate(ratios):
            while len(ratios[i]) < max_peaks:
                ratios[i].append(-10000000)
        for i, lst in enumerate(mc_ratios):
            while len(mc_ratios[i]) < max_peaks:
                mc_ratios[i].append(-10000000)
        X = np.column_stack((X_0, X_1, ratios, mc_ratios))
    else:
        X = np.column_stack((X_0, X_1))
    X_norm = scaler.transform(X)

    preds = model.predict_proba(X_norm)

    target_decoder = normalize_encoder(get_RF_encoder(reduced_set, include_chargestates, include_molecules))

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

    return elements, second_elements, confs, second_confs


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