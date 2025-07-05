# CrashPredictionExtension
This repository is based on the crash prediction pipeline and data from the following repository: https://github.com/Ludivine388/Crash-Prediction. The scenarios involve a pedestrian crossing the street while a bike with an onboard unit is approaching. The scene is filmed from a third-person view from the side of the street. The onboard unit is collecting V2X data from the bike. The data is brought into a format readable by the Atlas-Benchmark (https://github.com/boschresearch/the-atlas-benchmark) and then evaluated using the included framework.

## Pipeline and Data Format

The crash prediction pipeline processes video frames to generate standardized pedestrian trajectory data. All the referenced files are from the `Trajectory Prediction` folder of https://github.com/Ludivine388/Crash-Prediction. Note that the repo also consists of the files to get the pictures from the .rosbag files. The steps are as follows:

### 1. Initial Data

- **Input:** A folder containing image frames (e.g., `Trajectory Prediction/test_data/2024-08-22-15-35-05_folder/`).
- The pipeline starts with a sequence of images (frames) extracted from a video recording, stored in a dedicated folder.

### 2. Depth Analysis

- **Component:** `Trajectory Prediction/get_depth/get_depth.py`
- **Method:** Utilizes a pre-trained Pix2Pix model, based on "Learning the Depths of Moving People by Watching Frozen People" for monocular depth estimation.
- **Output:** For each input image, a corresponding depth image is generated and saved in `Trajectory Prediction/test_data/depthImg/{frame_folder}/`.

### 3. Pose Calculation

- **Component:** `Trajectory Prediction/get_pose/get_pose.py`
- **Method:** Uses a pre-trained OpenPose model (`body_pose_model.pth`) for multi-person 2D pose estimation, identifying 18 keypoints per person.
- **Output:**
  - Binary pose data files (`.pose`) for each frame in `Trajectory Prediction/test_data/poseData/pose/{frame_folder}/`.
  - Visualization images with poses drawn, saved in `Trajectory Prediction/test_data/poseData/img/{frame_folder}/`.

### 4. Tensor Generation

- **Component:** `Trajectory Prediction/get_tensor/generateTensor_Kalman.py`
- **Method:**
  - Extracts 2D keypoints from pose data.
  - Looks up depth values from the depth map.
  - Applies a Kalman filter to smooth depth estimates.
  - Projects 2D coordinates and smoothed depth into 3D (X, Y, Z) relative to the camera.
- **Output:** A single tensor file for all keypoints, saved in `Trajectory Prediction/test_data/InitTensor/Kalman/{frame_folder}/` via `joblib.dump`.

### 5. Relative Displacement Calculation

- **Component:** `Trajectory Prediction/get_gps_trajectory/get_ped_gps.py`
- **Method:**
  - Loads the tensor file.
  - Uses camera intrinsics to convert coordinates and depth into real-world (X, Y, Z) using `pixel_to_real`.
  - Calculates the average (X, Y) displacement per pedestrian per frame.
- **Output:** Dictionary mapping frame numbers to average (X, Y) displacement, saved as `ped_gps_per_frame` in `Trajectory Prediction/test_data/pedestrian_gps/{frame_folder}/`.

### 6. Formatting Coordinates

- **Component:** `format_coordinates`
- **Method:**
  - Iterates through `ped_gps_per_frame` files.
  - Reads frame ID and (X, Y) coordinates.
  - Converts each entry into JSON format:  
    ```json
    {
      "track": {
        "f": frame_id,
        "p": pedestrian_id,
        "x": x_coord,
        "y": y_coord
      }
    }
    ```
- **Output:** `_atlas.json` file for each input folder, saved in `Trajectory Prediction/test_data/atlas_json/mmcp/`, with one JSON object per line.

# MMCP Dataset and Prediction Models

The files are copied into a cloned version of the atlas-benchmark into the designated folder. Note that each scenario in the data will be in a `.njson`-File. In addition, a config is created to read the dataset into the benchmark classes. The current version of the mmcp-config (which stands for Multi-Modal-Crash-Prediction) is based on the config for the ETH-Dataset, which shows students on the campus of ETH University.

## 1. Configuration of the MMCP Dataset

A new configuration file, `dataset_config_mmcp.yaml`, has been created to work with the MMCP data. This file controls how the raw data is loaded, preprocessed, and split for experiments.

```yaml
# filepath: c:\Dokumente\Studium\HiWi\code\Atlas\the-atlas-benchmark\cfg\dataset_config_mmcp.yaml
# ...
# This branch collects all parameters related to the benchmark and experiment setup
benchmark:
  setup:
    interpolate: True
    smooth: 5 # 0 for no smoothing
    downsample: False
    downsample rate: 1 # fps  1 for no downsampling
    downsample map: False
    downsample map rate: 1
    added_noise_sigma: 0 # 0 for no added noise

    observation period: 6
    prediction horizon: 3
```

### Key Configuration Parameters:

*   **`dataset:`**: Defines the name (`mmcp`), the paths to the `.json` data files, and the recording frequency (`frequency: 2.5` Hz).
*   **`benchmark: setup:`**: Contains parameters for preprocessing and the experiment setup.
    *   `interpolate: True`: Missing positions in the trajectories are linearly interpolated.
    *   `smooth: 5`: Applies a moving average filter with a window size of 5 to the trajectories to reduce noise.
    *   `observation period: 6`: The number of past time steps (frames) given to the model as input (observation).
    *   `prediction horizon: 3`: The number of future time steps (frames) that the model is supposed to predict.

## 2. The "Rolling Prediction" Approach

The framework uses a "Rolling Prediction" (also known as "Sliding Window") approach to generate individual test scenarios from the continuous trajectory data.

Based on the current configuration (`observation period: 6`, `prediction horizon: 3`), this works as follows:
1.  A window with a total size of 9 (6 + 3) is slid over each agent's trajectory.
2.  For each position of the window, the **first 6 data points** are used as the **observation (input)** for the predictor.
3.  The **following 3 data points** serve as the **Ground Truth (GT)**, which is the correct future trajectory used to evaluate the model's prediction.

This process is repeated for all possible starting points along the trajectories, generating a large number of test scenarios for training and evaluation.
In addition, this mimics a real-time prediction approach, assuming we are always at time point 6 and look 3 points into the future.

## 3. Analysis in the Notebooks

The generated scenarios are used in the Jupyter notebooks to evaluate various prediction models.

### Evaluation Metrics

The framework uses the following metrics to assess the accuracy of the predictions:

*   **ADE (Average Displacement Error):** The average L2 distance between the entire predicted trajectory and the ground truth trajectory. It measures the average error across the whole prediction horizon.
*   **FDE (Final Displacement Error):** The L2 distance between the predicted final position and the ground truth final position. It specifically measures the error at the end of the prediction horizon.
*   **kADE / kFDE (Minimum ADE/FDE over k Samples):** These metrics are used for multi-modal predictors (like Trajectron++) that generate *k* possible future trajectories.
    *   **kADE (minADE):** The ADE of the best-predicted trajectory (the one closest to the ground truth) out of the *k* generated samples.
    *   **kFDE (minFDE):** The FDE of the best-predicted trajectory out of the *k* generated samples.

### Analysis in Notebooks

#### `understanding_benchmark_experiments.ipynb`

This notebook is used for evaluating and comparing "classic" (non-learning-based) prediction models.

**Used Classes:**
*   `Dataset`: Loads and processes the MMCP data according to `dataset_config_mmcp.yaml`.
*   `Predictor_kara`, `Predictor_sof`, `Predictor_zan`: Implementations of baseline models like Constant Velocity (CVM) and Social Forces.
*   `Benchmark`: Executes the experiments. It iterates over test scenarios, calls the `predict` method of a predictor, and calculates the evaluation metrics (ADE, FDE).
*   `Evaluator`: Used for visualizing scenarios, ground-truth trajectories, and model predictions.

#### `understanding_prediction_with_trajectonpp.ipynb`

This notebook focuses on the evaluation of a more complex, learning-based model.

**Used Classes:**
*   `TrajectronPredictor`: A wrapper for the **Trajectron++** model. Trajectron++ is a graph-based recurrent neural network (RNN) that predicts multi-modal (i.e., several possible) future trajectories. Its performance is evaluated using ADE/FDE for the most likely prediction and kADE/kFDE to assess the quality of its overall set of predictions.
### General model overview (Excel)
### Prediction mode
### CVM
### Trajectron ++

# Current problems

### Dataset availability
### Accuracy of Pipeline

# Possible Extensions
