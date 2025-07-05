import joblib
import numpy as np
from geopy.distance import geodesic
import os

script_dir = os.path.dirname(os.path.abspath(__file__))


#setting the return coordinate type
return_gps = False  # Set to True for GPS coordinates, False for relative displacement

for frame_folder in os.listdir(os.path.join(script_dir, f'../test_data/InitTensor/Kalman')):
    TensorPath = os.path.join(script_dir, f'../test_data/InitTensor/Kalman/{frame_folder}/Tensor')

    # outut path
    save_Ped_gps = os.path.join(script_dir, f"../test_data/pedestrian_gps/{frame_folder}/")
    os.makedirs(save_Ped_gps ,exist_ok=True)

    tensor = joblib.load(TensorPath)

    # Intrinsic parameters of RealSense D435i
    fx = 322.282410
    fy = 322.282410
    # Principal point x,y-coordinate (image width, height /2)
    cx = 320   
    cy = 240   

    # Known GPS coordinates of ARI
    lat_robot = 49.009298
    lon_robot = 8.409649

    # Convert pixel coordinates to world coordinates
    def pixel_to_real(u, v, depth, fx, fy, cx, cy):
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth  
        return np.array([X, Y, Z])

    # Convert world coordinates to GPS coordinates
    def convert_to_gps(X, Y, camera_gps, scale):
        distance_north = -Y * scale  
        distance_east = X * scale  

        #return relative displacement in meters
        if not return_gps:
            return distance_north, distance_east

        # Calculate new GPS location based on the displacement
        new_gps = geodesic(meters=distance_north).destination((camera_gps[0], camera_gps[1]), bearing=0)
        new_gps = geodesic(meters=distance_east).destination((new_gps.latitude, new_gps.longitude), bearing=90)
        return new_gps.latitude, new_gps.longitude

    # Processing each frame in the Tensor
    output_gps_per_frame = {}

    # Scale robot
    # Testing with frame 126 of 2024-08-22-15-35-05_folder, where pedestrian and eBike are on the same point
    correct_gps = (49.00935, 8.409685)
    calculated_gps = (49.00929854416467, 8.409863033065685)    
    real_distance =geodesic(calculated_gps, correct_gps).meters
    u_frame_126 = 517
    v_frame_126 = 238
    depth_frame_126 = 95.89230011834508
    X_frame_126, Y_frame_126, Z_frame_126 = pixel_to_real(u_frame_126, v_frame_126, depth_frame_126, fx, fy, cx, cy)

    # Calculate the camera world distance (in meters)
    camera_distance = np.sqrt(X_frame_126**2 + Y_frame_126**2 + Z_frame_126**2)

    # Calculate the scale
    scale_robot = real_distance / camera_distance
    print(scale_robot)

    for ped_data in tensor:
        # Check if empty values for this frame 
        if all(depth == 0 and u == 0 and v == 0 for _, depth, u, v in ped_data):
            print(f'No pedestrian found in frame:{ped_data[0][0]}')
            continue  # Skip for frame

        # Get the frame number
        frame_num = ped_data[0][0]

        # Calculate GPS locations
        gps_locations = []
        for (num, depth, u, v) in ped_data:
            # Convert pixel coordinates and depth to world coordinates
            X, Y, Z = pixel_to_real(u, v, depth, fx, fy, cx, cy)
            # Scale_robot calcul test:
            #if num == 126:
            #    print(depth)
            # Convert world coordinates to GPS coordinates
            gps_lat, gps_lon = convert_to_gps(X, Y, (lat_robot, lon_robot), scale_robot)

            # Add GPS location to the list for this pedestrian
            gps_locations.append((gps_lat, gps_lon))

        # Calculate the average GPS location per pedestrian
        avg_gps_lat = np.mean([loc[0] for loc in gps_locations])
        avg_gps_lon = np.mean([loc[1] for loc in gps_locations])
        
        # Store the result per frame
        output_gps_per_frame[frame_num] = (avg_gps_lat, avg_gps_lon)
    joblib.dump(output_gps_per_frame, save_Ped_gps+"ped_gps_per_frame")
    # Print and save the output
    for frame, gps_location in output_gps_per_frame.items():
        print(f"Frame {frame}: Average GPS location: {gps_location}")
