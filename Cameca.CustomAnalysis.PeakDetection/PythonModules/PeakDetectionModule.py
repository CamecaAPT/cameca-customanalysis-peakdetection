import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pkg_resources
import torch
from cameca_apt.extension import host, registry
from cameca_apt.extension.observablity import log_call
from peak_detection.RangingNN.predictor import DetectionPredictor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

from apt_loader import build_histogram, load_apt
from config import temp_predition_config_file
from models import EntryReturn
from training import ensure_training_data

logger = logging.getLogger(__name__)

# Match for splitting ion formulas into component elements: matches right before any uppercase character
# Lookbehind requires one character before, blocking matching at the beginning of the text, causing the
# split to not consider the beginning a match and split which will otherwise include an initial empty string
pattern = re.compile(r"(?<=.)(?=[A-Z])")

BIN_WIDTH = 0.01


@registry.register()
def preload_training_data(progressToken: str | int):
    ensure_training_data(progressToken)


@registry.register()
def predict_ranges(
    element_list: list[str],
    elements_to_get_molecules: list[str],
    confidence: float,
    intersectionOverUnion: float,
    maxDetections: int,
    iterations: int,
):
    training = ensure_training_data()
    ions = training.ions
    full_ion_counts = training.counts
    mc = training.mc
    encoder = training.encoder
    decoder = training.decoder

    mass_data = cast(npt.NDArray[np.float32], host.section_data("Mass"))
    histogram = build_histogram(mass_data, BIN_WIDTH, 0, 30720)
    spectrum_log_tensor = load_apt(histogram)[1]

    extension_dir = Path.cwd()
    save_dir = str(extension_dir)

    modelpath = pkg_resources.resource_filename("peak_detection", "RangingNN/modelweights/best_v0_2025-11-12.pt")

    prediction_args = {
        "conf": confidence,
        "iou": intersectionOverUnion,
        "max_det": maxDetections,
    }
    with temp_predition_config_file(prediction_args) as config_path:
        logger.debug("Temporary config path created at: %s", config_path)
        predictor = DetectionPredictor(
            modelpath,
            spectrum_log_tensor[None, None, ...],
            save_dir=save_dir,
            cfg=config_path,  # type: ignore -- Incomplete typing for cfg
        )
        # DetectionPredictor is not fully typed -- Cast result to continue type propogation here
        x = predictor()
        result = cast(list[torch.Tensor], x)[0]
        peak_pred = result[:, :2].cpu()
        counts = result[:, 2].cpu()
    logger.info("peak_pred.size(dim=0) = %d", peak_pred.size(dim=0))
    peak_iter = [1] * peak_pred.size(dim=0)
    logger.info("peak_iter: %s", peak_iter)

    # Multiple iterations
    for iteration in range(iterations - 1):
        peak_range_toadd: list[list[float]] = list()
        counts_toadd = list()
        multiplier = 0.01

        # Processing spectrum part 1
        n = spectrum_log_tensor.shape[0]
        x1 = np.arange(n) * multiplier  # equivalent to your linspace
        # peak_range_pred: list of (start, end)
        ranges = np.asarray(peak_pred, dtype=float)  # shape (R, 2)
        starts = ranges[:, 0] * multiplier  # shape (R,)
        ends = ranges[:, 1] * multiplier  # shape (R,)

        # Broadcasting: shape (n, R), then reduce along R
        in_any_range = np.logical_or.reduce((x1[:, None] > starts[None, :]) & (x1[:, None] < ends[None, :]), axis=1)

        idx_delete = x1[in_any_range]

        # Processing spectrum part 2
        x1 = np.asarray(x1)
        spectrum_log = np.asarray(spectrum_log_tensor)
        mask = np.isin(x1, idx_delete)

        spectrum_log_mod = spectrum_log.copy()
        spectrum_log_mod[mask] = 0.2

        spectrum_log_mod = torch.Tensor(spectrum_log_mod)

        # Running predictions
        with temp_predition_config_file(prediction_args) as config_path:
            predictor_mod = DetectionPredictor(
                modelpath,
                spectrum_log_mod[None, None, ...],
                save_dir=save_dir,
                cfg=config_path,  # type: ignore -- Incomplete typing for cfg
            )
            result_mod = predictor_mod()[0]
            peak_range_pred_mod = result_mod[:, :2].cpu()
            counts_mod = result_mod[:, 2].cpu()

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

                if multiplier * abs(float(start) - float(start2)) < min_dist:
                    min_dist = multiplier * abs(float(start) - float(start2))
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
        peak_iter += [iteration + 2] * len(peak_range_toadd)

    # Apply element filtering to cached full element data
    df = create_filtered_dataframe(
        mc,
        full_ion_counts,
        ions,
        encoder,
        element_list,
        elements_to_get_molecules,
    )

    # Train the model
    scaler, model = train_RF_model(df, n_estimators=25, random_state=42)
    # logger.info("model=%s", str(model))

    elem1, elem2, conf1, conf2, _ = run_RF_model(
        peak_pred,
        counts,
        scaler,
        model,
        decoder,
        use_peak_maxima=True,
        spectrum_log=np.asarray(spectrum_log_tensor),
    )

    retVal = EntryReturn(
        (peak_pred * BIN_WIDTH).tolist(), elem1, list(map(float, conf1)), elem2, list(map(float, conf2)), peak_iter
    )
    return retVal


@registry.register()
def recommend_elements(
    peak_range_pred: list[list[float]],
    elements: list[str],
    thresh: float,
    num_elements: int,
) -> list[str]:
    training = ensure_training_data()
    ions = training.ions
    counts = training.counts
    mc = training.mc
    encoder = training.encoder
    decoder = training.decoder

    encoded_elem = [encoder[x] for x in elements]

    pred_arr = np.asarray(peak_range_pred, dtype=np.float64) / 0.01

    element_only = filter_to_elements(encoder.keys())
    merged = list(set(elements).union(set(element_only)))
    df = create_filtered_dataframe(mc, counts, ions, encoder, merged)
    res = suggest_additional_elements(df, pred_arr, encoded_elem, thresh, num_elements)
    return list(decoder[x] for x in res)


def run_RF_model(
    peak_range_pred,
    counts,
    scaler,
    model,
    decoder,
    use_peak_maxima,
    spectrum_log: npt.NDArray,
):
    multiplier = 0.01

    if not use_peak_maxima:
        X_0 = multiplier * peak_range_pred[:, 0]
    else:
        # Need to use the predicted peak ranges and data file to get the m/c position of the peak maximum
        X_0 = list()
        for i in range(peak_range_pred.shape[0]):
            start = multiplier * peak_range_pred[i, 0]
            end = multiplier * peak_range_pred[i, 1]
            peak_max = 0
            max_ind = 0
            for count, j in enumerate([(multiplier * i) + (multiplier / 2) for i in range(0, int(307.2 / multiplier))]):
                if j >= start and j <= end:
                    if spectrum_log[count] > peak_max:
                        peak_max = spectrum_log[count]
                        max_ind = j
            # print('Have peak max', peak_max, 'at max ind', max_ind, 'for peak range', start, end)
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
            second_elements.append("NaN")
        second_confs.append(second_conf)

    return elements, second_elements, confs, second_confs, X


def train_RF_model(df, n_estimators=25, random_state=42):
    features = [col for col in df.columns if col not in ["ions", "target"]]

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    y = df["target"]
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)
    return scaler, model


def filtered_encoded(elem: list[str], mol: list[str], encoder: dict[str, int]):
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


def filter_to_elements(names: Iterable[str]) -> list[str]:
    """Filters a list of ion names to only elements by removing all complex ions"""
    elements: list[str] = []
    for ion in names:
        if not pattern.search(ion):
            elements.append(ion)
    return elements


@log_call(logger)
def create_filtered_dataframe(
    mc: npt.NDArray[np.float64],
    counts: npt.NDArray[np.float64],
    ions: npt.NDArray[np.int32],
    encoder: dict[str, int],
    element_list: list[str],
    elements_to_get_molecules: list[str] | None = None,
):
    elements_to_get_molecules = [] if elements_to_get_molecules is None else elements_to_get_molecules
    matches = filtered_encoded(element_list, elements_to_get_molecules, encoder)
    mask = np.isin(ions, matches)
    return pd.DataFrame(
        {
            "mc": mc[mask],
            "counts": counts[mask],
            "target": ions[mask],
        }
    )


def create_kde(series):
    return KernelDensity(kernel="gaussian", bandwidth=0.5).fit(series.array.reshape(-1, 1))


def make_lookup_model(element_df: pd.DataFrame):
    # Group by "target" column (the encoded elements), then for each grouped series of mc values,
    # create a KernalDensity instance to return
    # Finally map as a dictionary for expected output shape
    return element_df.groupby("target")["mc"].apply(create_kde).to_dict()


def suggest_additional_elements(
    elements_df: pd.DataFrame,
    peak_range_pred: npt.NDArray[np.float64],
    elements: list[int],
    thresh: float,
    num_elements: int,
) -> set[int]:
    """Returns set of encoded element ions that are good candidates for adding to the element list"""
    # Lookup model is then created from full training data of all basic elements
    lookup_model = make_lookup_model(elements_df)

    full_adds: set[int] = set()
    for pred, element in zip(peak_range_pred, elements):
        conf = np.exp(lookup_model[element].score_samples(pred.reshape(-1, 1) / 100))[0]

        if conf < thresh:
            add_ions = lookup_add_ions(
                lookup_model=lookup_model,
                x=pred[0].reshape(-1, 1) / 100,
                thresh=thresh,
            )
            for a in add_ions[:num_elements]:
                full_adds.add(a)
    return full_adds


def lookup_add_ions(lookup_model, x, thresh) -> list[int]:
    ranking_dict = dict()
    for k in lookup_model.keys():
        if k not in ranking_dict.keys():
            log_prob = lookup_model[k].score_samples(x)  # log(f(x0))
            pdf_value = np.exp(log_prob)[0]
            ranking_dict[k] = pdf_value

    prediction_rankings = sorted(ranking_dict.items(), key=lambda x: x[1], reverse=True)
    return [i for i, c in prediction_rankings if c > thresh]


def calculate_iou_1d(interval1: list[float], interval2: list[float]) -> float:
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
