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

## 3. Analysis and Evaluation

The generated scenarios are used in the Jupyter notebooks to evaluate and compare various prediction models using a set of standard metrics.

### Evaluation Metrics

The framework uses the following metrics to assess the accuracy of the predictions:

*   **ADE (Average Displacement Error):** The average L2 distance between the entire predicted trajectory and the ground truth trajectory. It measures the average error across the whole prediction horizon.
*   **FDE (Final Displacement Error):** The L2 distance between the predicted final position and the ground truth final position. It specifically measures the error at the end of the prediction horizon.
*   **kADE / kFDE (Minimum ADE/FDE over k Samples):** These metrics are used for multi-modal predictors (like Trajectron++) that generate *k* possible future trajectories.
    *   **kADE (minADE):** The ADE of the best-predicted trajectory (the one closest to the ground truth) out of the *k* generated samples.
    *   **kFDE (minFDE):** The FDE of the best-predicted trajectory out of the *k* generated samples.

### Analysis in Notebooks

#### `understanding_benchmark_experiments.ipynb`

This notebook serves as the primary tool for evaluating and comparing traditional, non-learning-based prediction models. It provides a clear baseline for model performance on the MMCP dataset.

**Core Classes and Their Roles:**

*   `Dataset`: This class is the foundation for data handling. It is responsible for loading the raw `.json` files specified in `dataset_config_mmcp.yaml`. During initialization, it applies the preprocessing steps defined in the configuration, such as interpolation and smoothing. Its `extract_scenarios` method implements the "Rolling Prediction" logic, creating the individual observation/ground-truth pairs used for evaluation.
*   `Predictor_kara`, `Predictor_sof`, `Predictor_zan`: These classes encapsulate classic trajectory prediction algorithms. They are initialized with the dataset and a method-specific configuration.
    *   `Predictor_kara`: Implements a **Constant Velocity Model (CVM)**. This is a simple kinematic model that extrapolates the agent's last known velocity to predict its future path. It serves as a fundamental baseline.
    *   `Predictor_sof`: Implements a **Social Forces Model**. This physics-based model treats agents as particles that are influenced by forces, including a driving force toward a goal and repulsive forces from other agents and obstacles. It models basic social interactions.
*   `Benchmark`: This class orchestrates the evaluation process. It takes a set of scenarios (from the `Dataset` object) and a predictor object. Its `accuracy_experiment` method iterates through each scenario, calls the predictor's `predict` method, and computes the specified metric (e.g., 'ade' or 'fde') by comparing the prediction to the ground truth.
*   `Evaluator`: This utility class is used for calculating the final error metrics and for visualization. It contains the logic to compute ADE and FDE from a prediction and a ground truth trajectory. Its `plot_scenario` method can be used to visually inspect a scene, including the agent's past path, the ground truth future, and the model's prediction.

#### `understanding_prediction_with_trajectonpp.ipynb`

This notebook is dedicated to evaluating a state-of-the-art, learning-based model, providing a comparison against the simpler baselines from the other notebook.

**Core Classes and Their Roles:**

*   `TrajectronPredictor`: This class acts as a wrapper for the powerful **Trajectron++** model. Trajectron++ is a pre-trained, graph-based recurrent neural network (RNN) designed for trajectory prediction.
    *   **Graph-Based Approach**: It models the scene as a dynamic graph where agents are nodes and their spatial proximity defines the edges. This structure allows the model to explicitly reason about social interactions between multiple agents simultaneously.
    *   **Multi-modality**: Unlike the classic models that produce a single deterministic output, Trajectron++ generates a distribution over several possible future trajectories (*k* samples). This captures the inherent uncertainty of human movement (e.g., a person at an intersection could go straight, turn left, or turn right).
    *   **Evaluation**: Because of its multi-modal nature, its performance is assessed not only with ADE/FDE on its most likely prediction but also with kADE/kFDE. These "minimum-over-k" metrics evaluate how well the set of all *k* predictions covers the actual future path, rewarding the model if at least one of its predictions was close to the ground truth.


### General model overview (Excel)
### Prediction mode
### CVM
### Trajectron ++

# Current problems

### Dataset availability
### Accuracy of Pipeline

# Possible Extensions
