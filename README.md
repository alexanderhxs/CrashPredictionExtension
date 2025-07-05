# CrashPredictionExtension
This repository is based on the crash prediction pipeline and data from the following repository: https://github.com/Ludivine388/Crash-Prediction. The scenarios involve a pedestrian crossing the street while a bike with an onboard unit is approaching. The scene is filmed from a third-person view from the side of the street. The onboard unit is collecting V2X data from the bike. The data is brought into a format readable by the Atlas-Benchmark (https://github.com/boschresearch/the-atlas-benchmark)

## Pipeline and Data Format

The crash prediction pipeline processes video frames to generate standardized pedestrian trajectory data. All the referenced files are from the 'Trajectory Prediction' folder of https://github.com/Ludivine388/Crash-Prediction. Note that the repo also consists of the files to get the pictures from the .rosbag files. The steps are as follows:

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



# Predictions and Results

### General model overview (Excel)
### Prediction mode
### CVM
### Trajectron ++

# Current problems

### Dataset availability
### Accuracy of Pipeline

# Possible Extensions
