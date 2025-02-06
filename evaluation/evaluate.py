"""
The following is the evaluation script used for the trackrad2025 challenge.

It is meant to run within a container on the Grand-Challenge.org platform.

To test is locally use the testing script at the root of the trackrad2025 repository.

This scripts implements 5 metrics:
- Dice similarity coefficient
- Hausdorff distance 95th percentile
- Average surface distance
- 2D center of mass error / euclidean center distance
- Dose error -> computed with a shifted point cloud approach
"""
import json
from glob import glob
import SimpleITK
import os
import datetime
from statistics import mean
from pathlib import Path
from pprint import pformat, pprint
from helpers import run_prediction_processing
import numpy as np
import scipy
import pandas as pd
import traceback
from typing import Any, Sequence

# import a custom reimplementation of the monai metrics
import monai_metrics as monai_metrics
TensorOrList = Sequence[np.ndarray] | np.ndarray

# region Define custom metrics not provided by monai

class EuclideanCenterDistanceMetric(monai_metrics.IterationMetric):
    def __init__(self) -> None:
        super().__init__()

    def _compute_tensor(self, y_pred: np.ndarray, y_true: np.ndarray | None = None, **kwargs):
        """
        Computation logic for `y_pred` and `y` of an iteration, the data should be "batch-first" Tensors.
        A subclass should implement its own computation logic.
        The return value is usually a "batch_first" tensor, or a list of "batch_first" tensors.
        """
        B,C,H,W = y_pred.shape
        true_com_path = np.array([scipy.ndimage.center_of_mass(y_true[i,0,:,:]) for i in range(B)])
        pred_com_path = np.array([scipy.ndimage.center_of_mass(y_pred[i,0,:,:]) for i in range(B)])

        # L2 norm of the difference between the predicted and true center of mass
        return np.linalg.norm(true_com_path - pred_com_path, axis=1)

# region dosimetric metrics helper functions
def shift_by_centroid_diff(pred_centroid, true_centroid, true_seg):
    """Shift input segmentation by difference between predicted and true centroids.

    Args:
        pred_centroid (list): predicted centroids in SI and AP
        true_centroid (list): true centroids in SI and AP
        true_seg (arr): true binary mask with shape (h,w)

    Returns:
        arr: shifted binary mask
    """
    
    # difference in centroids positions for 250 ms forecast
    delta_centroids_SI = pred_centroid[0][0] - true_centroid[0][0]  
    delta_centroids_AP = pred_centroid[0][1] - true_centroid[0][1] 
    
    # take last input segmentation and shift it by delta centroids  
    shifted_seg = scipy.ndimage.shift(true_seg, 
                                    shift=[delta_centroids_SI, delta_centroids_AP],
                                    order=3, mode='nearest')

    return shifted_seg

def calculate_dvh(dose_arr, label_arr, bins=1001):
    """Calculates a dose-volume histogram.
    Adapted from https://github.com/pyplati/platipy/blob/master/platipy/imaging/dose/dvh.py

    Args:
        dose_arr (numpy.ndarray): The dose grid.
        label_arr (numpy.ndarray): The (binary) label defining a structure.
        bins (int | list | np.ndarray, optional): Passed to np.histogram,
            can be an int (number of bins), or a list (specifying bin edges). Defaults to 1001.

    Returns:
        bins (numpy.ndarray): The points of the dose bins
        values (numpy.ndarray): The DVH values
    """

    # check that dose and label array have the same shape
    if dose_arr.shape != label_arr.shape:
        raise ValueError("Dose grid size does not match label, please resample!") 

    # Get dose values for the structure
    dose_vals = dose_arr[np.where(label_arr)]

    # Calculate the histogram
    counts, bin_edges = np.histogram(dose_vals, bins=bins)

    # Get mid-points of bins
    bins = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    # Calculate the actual DVH values
    values = np.cumsum(counts[::-1])[::-1]
    
    if np.all(values == 0):
        return bins, values
    else:
        values = values / values.max()
        return bins, values

def calculate_dvh_for_labels(dose_array, labels, bin_width=0.1, max_dose=None, spacing=(1,1,1)):
    """Calculate the DVH for multiple labels.
    Adapted from https://github.com/pyplati/platipy/blob/master/platipy/imaging/dose/dvh.py

    Args:
        dose_array (np.ndarray): Dose grid
        labels (dict): Dictionary of labels with the label name as key and binary mask np.ndarray as
            value.
        bin_width (float, optional): The width of each bin of the DVH (Gy). Defaults to 0.1.
        max_dose (float, optional): The maximum dose of the DVH. If not set then maximum dose from
            dose grid is used.Defaults to None.
        spacing (tuple, optional): The voxel spacing of the dose grid. Defaults to (1,1,1). Should be the same as for the masks!

    Returns:
        pandas.DataFrame: The DVH for each structure along with the mean dose and size in cubic
            centimetres as a data frame.
    """

    dvh = []

    label_keys = labels.keys()

    if not max_dose:
        max_dose = dose_array.max()

    for k in label_keys:
        # Get mask from dict
        mask_array = labels[k]

        # Compute cubic centimetre volume of structure
        cc = mask_array.sum() * np.prod([a / 10 for a in spacing])

        bins, values = calculate_dvh(
            dose_array, mask_array, bins=np.arange(-bin_width / 2, max_dose + bin_width, bin_width)
        )

        # Remove rounding error
        bins = np.round(
            bins.astype(float),
            decimals=10,
        )

        mean_dose = dose_array[mask_array > 0].mean()
        entry = {
            **{
                "label": k,
                "cc": cc,
                "mean": mean_dose,
            },
            **dict(zip(bins, values)),
        }

        dvh.append(entry)

    return pd.DataFrame(dvh)

def calculate_d_x(dvh, x, label=None):
    """Calculate the dose which x percent of the volume receives

    Args:
        dvh (pandas.DataFrame): DVH DataFrame as produced by calculate_dvh_for_labels
        x (float|list): The dose threshold (or list of dose thresholds) which x percent of the
            volume receives
        label (str, optional): The label to compute the metric for. Computes for all metrics if not
            set. Defaults to None.

    Returns:
        pandas.DataFrame: Data frame with a row for each label containing the metric and value.
    """

    if label:
        dvh = dvh[dvh.label == label]

    if not isinstance(x, list):
        x = [x]

    bins = np.array([b for b in dvh.columns if isinstance(b, float)])
    values = np.array(dvh[bins])

    metrics = []
    for idx in range(len(dvh)):
        d = dvh.iloc[idx]

        m = {"label": d.label}

        for threshold in x:
            value = np.interp(threshold / 100, values[idx][::-1], bins[::-1])
            if values[idx, 0] == np.sum(values[idx]):
                value = 0

            # Interp will return zero when computing D100, do compute this separately
            if threshold == 100:
                i, j = np.where(values == 1.0)
                value = bins[j][i == idx][-1]

            m[f"D{threshold}"] = value

        metrics.append(m)

    return pd.DataFrame(metrics)

def calculate_v_x(dvh, x, label=None):
    """Get the volume (in cc) which receives x dose

    Args:
        dvh (pandas.DataFrame): DVH DataFrame as produced by calculate_dvh_for_labels
        x (float|list): The dose threshold (or list of dose thresholds) to get the volume for.
        label (str, optional): The label to compute the metric for. Computes for all metrics if not
            set. Defaults to None.

    Returns:
        pandas.DataFrame: Data frame with a row for each label containing the metric and value.
    """

    if label:
        dvh = dvh[dvh.label == label]

    if not isinstance(x, list):
        x = [x]

    bins = np.array([b for b in dvh.columns if isinstance(b, float)])
    values = np.array(dvh[bins])

    metrics = []
    for idx in range(len(dvh)):
        d = dvh.iloc[idx]

        m = {"label": d.label}

        for threshold in x:
            value = np.interp(threshold, bins, values[idx]) * d.cc

            metric_name = f"V{threshold}"
            if threshold - int(threshold) == 0:
                metric_name = f"V{int(threshold)}"

            m[metric_name] = value

        metrics.append(m)

    return pd.DataFrame(metrics)

def calculate_d_cc_x(dvh, x, label=None):
    """Compute the dose which is received by cc of the volume

    Args:
        dvh (pandas.DataFrame): DVH DataFrame as produced by calculate_dvh_for_labels
        x (float|list): The cc (or list of cc's) to compute the dose at.
        label (str, optional): The label to compute the metric for. Computes for all metrics if not
            set. Defaults to None.

    Returns:
        pandas.DataFrame: Data frame with a row for each label containing the metric and value.
    """

    if label:
        dvh = dvh[dvh.label == label]

    if not isinstance(x, list):
        x = [x]

    metrics = []
    for idx in range(len(dvh)):

        d = dvh.iloc[idx]
        m = {"label": d.label}

        for threshold in x:
            cc_at = (threshold / dvh[dvh.label == d.label].cc.iloc[0]) * 100
            cc_at = min(cc_at, 100)
            cc_val = calculate_d_x(dvh[dvh.label == d.label], cc_at)[f"D{cc_at}"].iloc[0]

            m[f"D{threshold}cc"] = cc_val

        metrics.append(m)

    return pd.DataFrame(metrics)

#endregion

class DoseErrorMetric(monai_metrics.IterationMetric):
    def __init__(self) -> None:
        super().__init__()

    def _compute_tensor(self, y_pred: np.ndarray, y_true: np.ndarray | None = None, sigma=4, **kwargs):
        """
        Computation logic for `y_pred` and `y` of an iteration, the data should be "batch-first" Tensors.
        A subclass should implement its own computation logic.
        The return value is usually a "batch_first" tensor, or a list of "batch_first" tensors.
        """
        # filter out empty targets
        empty_targets = np.sum(y_true, axis=(1,2,3)) == 0
        y_pred = y_pred[~empty_targets]
        y_true = y_true[~empty_targets]

        B,C,H,W = y_pred.shape
        
        initial_target_mask = y_true[0,0,:,:]

        # add an isotropic 3 pixel margin to the moving seg
        expanded_initial_target_mask = scipy.ndimage.binary_dilation(initial_target_mask, structure=scipy.ndimage.generate_binary_structure(2, 1), iterations=3).astype(initial_target_mask.dtype)
        
        #expanded_initial_target_mask = (scipy.ndimage.gaussian_filter(expanded_initial_target_mask, sigma=sigma)> 0.0).astype(initial_target_mask.dtype) # sigma=4 or 6
    
        labels = {"GTV": initial_target_mask}
        d98_original = calculate_d_x(calculate_dvh_for_labels(expanded_initial_target_mask, labels), [2, 98]).loc[0, 'D98']

        targets_com = np.array([scipy.ndimage.center_of_mass(y_true[i,0,:,:]) for i in range(B)])
        outputs_com = np.array([scipy.ndimage.center_of_mass(y_pred[i,0,:,:]) for i in range(B)])
    
        # array in which to store final summed up shifted doses for current patient
        final_shifted_dose = np.zeros(shape=(H,W), dtype=np.float32)
        
        for i in range(B):
            # shift by centroid difference and sum up current dose to final one
            final_shifted_dose += shift_by_centroid_diff(outputs_com[i:i+1], targets_com[i:i+1], expanded_initial_target_mask)
        
        # divide by number of shifts to normalize dose
        final_shifted_dose /= B

        # Dose diff
        # dose_diff = final_shifted_dose - expanded_initial_target_mask

        # Compute DVH for final dose
        d98_final = calculate_d_x(calculate_dvh_for_labels(final_shifted_dose, labels), [2, 98]).loc[0, 'D98']

        # get realtive decrease of d98
        relative_d98_decrease = (d98_original - d98_final) * 100 / d98_original
        return np.array((relative_d98_decrease,))

#endregion

# Define of the metrics used
dice_metric = monai_metrics.DiceMetric()
surface_distance_95_metric = monai_metrics.HausdorffDistanceMetric(percentile=95)
surface_distance_avg_metric = monai_metrics.SurfaceDistanceMetric()
center_of_mass_error_metric = EuclideanCenterDistanceMetric()
dose_error_metric = DoseErrorMetric()

# Helper function to extract the runtime of a job
def extract_runtime(job):
    completed_at = datetime.datetime.fromisoformat(job["completed_at"])
    started_at = datetime.datetime.fromisoformat(job["started_at"])
    return (completed_at - started_at).total_seconds()

# directories populated by GC
INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")

# use the ground truth directory populated by GC or the local one if it does not exist
GROUND_TRUTH_DIRECTORY = Path("/opt/ml/input/data/ground_truth/") if os.path.exists("/opt/ml/input/data/ground_truth/") else Path("ground_truth")

def process(job):
    """Processes a single algorithm job, looking at the outputs"""
    
    report = "Processing:\n"
    report += pformat(job)
    report += "\n"

    # Firstly, find the location of the results
    location_mri_linac_series_targets = get_file_location(
            job_pk=job["pk"],
            values=job["outputs"],
            slug="mri-linac-series-targets",
        )
    
    # Secondly, read the results
    result_mri_linac_series_targets = load_image_file(
        location=location_mri_linac_series_targets,
    )

    # Thirdly, retrieve the input file name to match it with your ground truth
    image_name_mri_linac_series = get_image_name(
            values=job["inputs"],
            slug="mri-linac-series",
    )
    image_name_mri_linac_target = get_image_name(
            values=job["inputs"],
            slug="mri-linac-target",
    )

    # Extract the case id from the image name
    case_id = os.path.basename(image_name_mri_linac_series).replace(".mha", "")
    print(case_id)

    # Fourthly, load the ground truth form the ground truth directory mounted by GC
    ground_truth_image = SimpleITK.ReadImage(GROUND_TRUTH_DIRECTORY / f"{case_id}/targets/{case_id}_labels.mha")

    # Convert the ground truth to a numpy array and transpose it to the correct shape (H,W,T) -> (T,H,W)
    ground_truth = SimpleITK.GetArrayFromImage(ground_truth_image)
    

    y_pred = result_mri_linac_series_targets
    y_true = ground_truth
    
    # Transpose the arrays to (T,H,W) format
    y_true = y_true.transpose(2,0,1)
    y_pred = y_pred.transpose(2,0,1)

    # Check if the shapes match
    assert y_true.shape == y_pred.shape, f"shapes {y_true.shape} and {y_pred.shape} do not match"

    T, H, W = y_true.shape

    # Convert to the shape required by monai (T,C,H,W) where C = 1 is the channel dimension
    y_pred_monai = y_pred.reshape(T,1,H,W)
    y_true_monai = y_true.reshape(T,1,H,W)

    # Finally, calculate by comparing the ground truth to the actual results

    # Dice similarity coefficient
    dsc = dice_metric(y_pred_monai, y_true_monai).mean().item()
    
    # Hausdorff distance 95th percentile
    surface_distance_95 = surface_distance_95_metric(y_pred_monai, y_true_monai).mean().item()

    # Average surface distance
    surface_distance_average = surface_distance_avg_metric(y_pred_monai, y_true_monai).mean().item()
    
    # 2D center of mass error
    com_error = center_of_mass_error_metric(y_pred_monai, y_true_monai).mean().item()

    # dosimetric metric
    try:
        dose_error = dose_error_metric(y_pred_monai, y_true_monai).mean().item()
    except Exception:
        # this is a fallback in case the dose error metric fails
        print(traceback.format_exc())
        dose_error = 0

    # Extract the runtime of the job for the runtime metric
    runtime = extract_runtime(job)

    # Return the metrics
    return {
        "case_id": case_id,
        "dice_similarity_coefficient": dsc,
        "surface_distance_95": surface_distance_95,
        "surface_distance_average": surface_distance_average,
        "com_error": com_error,
        "dose_error": dose_error,
        "total_runtime": runtime,
        "frame_count": T,
    }

def process_with_error(job):
    try:
        return process(job)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

def main():
    print_inputs()

    metrics = {}
    predictions = read_predictions()

    # We now process each algorithm job for this submission
    # Note that the jobs are not in any order!
    # We work that out from predictions.json

    # Use concurrent workers to process the predictions more efficiently
    metrics["results"] = run_prediction_processing(fn=process_with_error, predictions=predictions)

    # Loading time estimation
    # We want to rank the algorithms based on the runtime of the algorithm.
    # However, in a clinical setting, the loading time of the model is not relevant.
    # Therefore, we exclude the loading time of the model.
    # One way to compute the loading time and time per frame is a linear regression model
    frame_counts = [case_metrics["frame_count"] for case_metrics in metrics["results"]]
    runtimes = [case_metrics["total_runtime"] for case_metrics in metrics["results"]]
    time_per_frame, loading_time = np.polyfit(frame_counts, runtimes, 1) 

    # We have the results per prediction, we can aggregate over the results and
    # generate an overall score(s) for this submission.
    metrics["aggregates"] = {
        # geometric
        "dice_similarity_coefficient": mean(result["dice_similarity_coefficient"] for result in metrics["results"]), # Dice similarity coefficient
        "surface_distance_95": mean(result["surface_distance_95"] for result in metrics["results"]), # Hausdorff distance 95th percentile
        "surface_distance_average": mean(result["surface_distance_average"] for result in metrics["results"]), # Average surface distance
        "com_error": mean(result["com_error"] for result in metrics["results"]), # 2D com error
        
        # dosimetric
        "dose_error": mean(result["dose_error"] for result in metrics["results"]), # Shifted point cloud approach
        
        # runtime
        "runtime": mean(result["total_runtime"] - loading_time for result in metrics["results"]), # mean runtime without loading time

        # other metadata which might be interesting for the evaluation
        "loading_time": loading_time,
        "time_per_frame": time_per_frame,
        "total_time" : sum(result["total_runtime"] for result in metrics["results"]) # total time for all predictions
    }

    # Make sure to save the metrics
    write_metrics(metrics=metrics)

    return 0


def print_inputs():
    # Just for convenience, in the logs you can then see what files you have to work with
    input_files = [str(x) for x in Path(INPUT_DIRECTORY).rglob("*") if x.is_file()]

    print("Input Files:")
    pprint(input_files)
    print("")


def read_predictions():
    # The prediction file tells us the location of the users' predictions
    with open(INPUT_DIRECTORY / "predictions.json") as f:
        return json.loads(f.read())


def get_image_name(*, values, slug):
    # This tells us the user-provided name of the input or output image
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    # Gets the location of the interface relative to the input or output
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["interface"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk, values, slug):
    # Where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_image_file(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))

    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def write_metrics(*, metrics):
    # Write a json document used for ranking results on the leaderboard
    with open(OUTPUT_DIRECTORY / "metrics.json", "w") as f:
        f.write(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    raise SystemExit(main())
