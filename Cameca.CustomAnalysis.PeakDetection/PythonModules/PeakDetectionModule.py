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

    if implementation == PeakDetectionImplementation.NeuralNetwork:
        res, confidence, profile_final = predict_peak_ions(spectrum, peak_pred, bin_width=0.01, max_width_Da=0.5)
        return peak_pred, res, confidence, None, None
    elif implementation == PeakDetectionImplementation.RandomForest:
        scaler_path = pkg_resources.resource_filename('randomforest', 'StandardScaler.pkl')
        model_path = pkg_resources.resource_filename('randomforest', 'RandomForestClassifier.pkl')
        elem1, elem2, conf1, conf2 = random_forest(peak_pred, counts, scaler_path, model_path)
        return peak_pred, elem1, conf1, elem2, conf2
    elif implementation == PeakDetectionImplementation.RandomForestReduced:
        scaler_path = pkg_resources.resource_filename('randomforest', 'StandardScaler_reducedset.pkl')
        model_path = pkg_resources.resource_filename('randomforest', 'RandomForestClassifier_reducedset.pkl')
        elem1, elem2, conf1, conf2 = random_forest(peak_pred, counts, scaler_path, model_path)
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
    

def get_RF_encoder():
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
    return target_decoder


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
    

def predict_peak_ions(spectrum, peak_range_pred, bin_width=0.01, max_width_Da=0.5):
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
    modelpath = pkg_resources.resource_filename('peak_detection', 'Ionclassifier/modelweights/model_bestepoch.tar')
    RNNmodel = torch.load(modelpath, map_location='cpu')['ema']
    le = get_label_encoder()
    res, confidence = predict_elements(RNNmodel, profile_final, le, 'cpu')

    return res, confidence, profile_final


def random_forest(peak_range_pred, counts, scaler_path, model_path):
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    X_0 = 0.01*peak_range_pred[:, 0]
    X_1 = counts
    X = np.column_stack((X_0, X_1))
    X_norm = scaler.transform(X)

    preds = model.predict_proba(X_norm)

    target_decoder = get_RF_encoder()

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

