# CrashPredictionExtension
This repository is based on the crash prediction pipeline and data from the following repository: https://github.com/Ludivine388/Crash-Prediction. The scenarios involve a pedestrian crossing the street while a bike with an onboard unit is approaching. The scene is filmed from a third-person view from the side of the street. The onboard unit is collecting V2X data from the bike. The data is brought into a format readable by the Atlas-Benchmark (https://github.com/boschresearch/the-atlas-benchmark)

# Pipeline and Data format

1. Initial Data
The pipeline starts with a sequence of images (frames) extracted from a video recording. These frames should be located in a dedicated folder.

Input: A folder containing image frames (e.g., Trajectory Prediction/test_data/2024-08-22-15-35-05_folder/).
2. Depth Analysis
This step generates a depth map for each input frame, estimating the distance of each pixel from the camera.

Component: Trajectory Prediction/get_depth/get_depth.py
Method: It utilizes a pre-trained Pix2Pix model, based on the work "Learning the Depths of Moving People by Watching Frozen People". The model performs monocular depth estimation on each frame. The required model checkpoints can be downloaded by running the fetch_checkpoints.sh script.
Output: For each input image, a corresponding depth image is generated and saved in Trajectory Prediction/test_data/depthImg/{frame_folder}/.
3. Pose Calculation
This step detects pedestrians in each frame and estimates their 2D body pose by identifying key body joints.

Component: Trajectory Prediction/get_pose/get_pose.py
Method: A pre-trained OpenPose model (body_pose_model.pth) is used to perform multi-person 2D pose estimation. It identifies 18 keypoints for each detected person.
Output:
Binary pose data files (.pose) are saved for each frame in Trajectory Prediction/test_data/poseData/pose/{frame_folder}/. These files contain the pixel coordinates of the detected keypoints for each person.
Visualization images with the drawn poses are saved in Trajectory Prediction/test_data/poseData/img/{frame_folder}/.
4. Tensor Generation
The 2D pose coordinates and depth maps are combined to compute the 3D world coordinates for each pedestrian keypoint relative to the camera.

Component: Trajectory Prediction/get_tensor/generateTensor_Kalman.py
Method:
For each detected person, the script extracts the 2D keypoint coordinates from the pose data.
It looks up the corresponding depth values from the generated depth map.
A Kalman filter is applied to the average depth value of a person's keypoints to smooth the depth estimation across frames, reducing noise and correcting outliers.
The 2D pixel coordinates (u, v) and the smoothed depth value (d) are projected into a 3D coordinate system (X, Y, Z) relative to the camera.
Output: A single Tensor file is created using joblib.dump and saved in Trajectory Prediction/test_data/InitTensor/Kalman/{frame_folder}/. This file contains a list of 3D coordinates for all keypoints of all detected pedestrians in every frame.
5. Relative Displacement Calculation
This step processes the 3D tensor to calculate the final relative displacement of each pedestrian from the camera's position.

Component: Trajectory Prediction/get_gps_trajectory/get_ped_gps.py
Method: Although the script name suggests GPS conversion, it is used here to extract relative world coordinates.
It loads the Tensor file generated in the previous step.
The script uses the camera's intrinsic parameters to convert the pixel coordinates and depth into real-world coordinates (X, Y, Z) using the pixel_to_real function. This represents the displacement in meters from the camera.
It calculates the average (X, Y) displacement for each pedestrian in every frame.
Output: A dictionary mapping each frame number to the pedestrian's average (X, Y) displacement. This is saved as a ped_gps_per_frame file in Trajectory Prediction/test_data/pedestrian_gps/{frame_folder}/.
6. Formatting Coordinates
The final step formats the calculated displacements into a standardized JSON format, with one JSON object per line.

Component: format_coordinates
Method:
The script iterates through the ped_gps_per_frame files generated in the previous step.
It reads the frame ID and the (X, Y) coordinate tuple for each entry.
Each entry is converted into a JSON object with the structure {"track":{"f":frame_id, "p":pedestrian_id, "x":x_coord, "y":y_coord}}.
Output: A _atlas.json file is created for each input folder and saved in Trajectory Prediction/test_data/atlas_json/mmcp/. Each line in the file is a JSON string representing a single point in a pedestrian's trajectory.



# Predictions and Results

### General model overview (Excel)
### Prediction mode
### CVM
### Trajectron ++

# Current problems

### Dataset availability
### Accuracy of Pipeline

# Possible Extensions
