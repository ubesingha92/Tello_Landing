import cv2
import numpy as np
from djitellopy import Tello
import math
import time
import datetime
from simple_pid import PID
import csv
import os

# Define frame size
FRAME_SIZE = (320, 240)

# Path to calibration data
calib_data_path = "Camera_Calibartion/calib.npz"

# Size of the ArUco markers (in centimeters)
MARKER_SIZE = 7.5  # cm

# Define the sample time
SAMPLE_TIME = 0.4  # seconds

# PID coefficients for X
# KP_X, KI_X, KD_X = 3, 0, 5
KP_X, KI_X, KD_X = 0, 0, 0
MAX_SPEED_X = 15
pid_x = PID(KP_X, KI_X, KD_X, setpoint=0, sample_time = SAMPLE_TIME)
pid_x.output_limits = (-MAX_SPEED_X, MAX_SPEED_X)

# PID coefficients for Y
# KP_Y, KI_Y, KD_Y = 3, 0, 5
KP_Y, KI_Y, KD_Y = 0, 0, 0
MAX_SPEED_Y = 15
pid_y = PID(KP_Y, KI_Y, KD_Y, setpoint=0, sample_time = SAMPLE_TIME)
pid_y.output_limits = (-MAX_SPEED_Y, MAX_SPEED_Y)

# PID coefficients for yaw
# KP_YAW, KI_YAW, KD_YAW = 4, 0.01, 1
KP_YAW, KI_YAW, KD_YAW = 4, 0.01, 1
MAX_SPEED_YAW = 15
pid_yaw = PID(KP_YAW, KI_YAW, KD_YAW, setpoint=0, sample_time = SAMPLE_TIME)
pid_yaw.output_limits = (-MAX_SPEED_YAW, MAX_SPEED_YAW)

XY_ACCEPT_RANGE = 3
YAW_ACCEPT_RANGE = 3

# Connect to Tello drone
tello = Tello()
tello.connect()

filename_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
csv_filename = f'csv/aruco_marker_errors_{filename_time}.csv'

def load_calibration_data(path):
    # Load calibration data
    calib_data = np.load(path)

    cam_mat = calib_data["camMatrix"]
    dist_coef = calib_data["distCoef"]

    return cam_mat, dist_coef

def write_error_to_csv(file_path, error_data):
    # Column titles
    column_titles = ['Time', 'Marker ID', 'Error X', 'Error Y', 'Yaw Error', 'Height', 'PID Value X', 'PID Value Y', 'PID Value Yaw']
    
    # Check if the file exists and is empty
    file_exists = os.path.isfile(file_path)
    write_header = not file_exists or os.stat(file_path).st_size == 0
    
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # If the file is new or empty, write the column titles first
        if write_header:
            writer.writerow(column_titles)
        
        # Then write the actual data
        writer.writerow(error_data)

def control_markers(frame, corners, ids, rvecs, tvecs, height, roll, pitch):

    for i in range(len(ids)):
        cv2.aruco.drawDetectedMarkers(frame, corners)

        x, y, _ = [int(round(item)) for sublist in tvecs[i] for item in sublist]  # Unpack and round tvecs[i]

        # Correction
        error_x = -x + int(height * math.sin(roll))
        error_y = y - int(height * math.sin(pitch))

        # Calculate Euler angles
        rotation_matrix, _ = cv2.Rodrigues(rvecs[i])
        euler_angles = cv2.decomposeProjectionMatrix(np.hstack((rotation_matrix, tvecs[i].reshape(3, 1))))[6]

        # Round angles and convert to integer (Roll, Pitch, Yaw)
        euler_angles = [int(round(angle[0])) for angle in euler_angles]
        yaw_error = euler_angles[2]

        # Get Battery
        battery_value = tello.get_battery()

        # PID Control
        pid_val_x = int(pid_x(error_x))
        pid_val_y = int(pid_y(error_y))
        pid_val_z = 0
        pid_val_yaw = int(pid_yaw(yaw_error))

        print (f"Battery: {battery_value} X: {error_x} Y: {error_y} Yaw: {yaw_error} Height: {height}", end="")

        # Tello movement control based on PID adjustments
        if -YAW_ACCEPT_RANGE > yaw_error or yaw_error > YAW_ACCEPT_RANGE:
            pid_val_x = 0
            pid_val_y = 0
            print ("--Need Yaw Correction")

        elif (-XY_ACCEPT_RANGE > error_x or error_x > XY_ACCEPT_RANGE) and (-XY_ACCEPT_RANGE > error_y or error_y > XY_ACCEPT_RANGE):
            print ("--Need XY Correction")

        else:
            print ("--Ok")
            # tello.land()

        # Write errors to CSV
        human_readable_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        error_data = [human_readable_time, ids[i][0], error_x, error_y, yaw_error, height, pid_val_x, pid_val_y, pid_val_yaw]
        write_error_to_csv(csv_filename, error_data)

        # RC Control
        # tello.send_rc_control(pid_val_x, pid_val_y, pid_val_z, pid_val_yaw)

def process_frame(tello, frame_size):
    # Read frame from the drone
    frame = tello.get_frame_read().frame

    # Rotate the frame and crop around the center
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = frame[frame.shape[0] - frame_size[0]:frame.shape[0],
            frame.shape[1] - frame_size[1]:frame.shape[1]]
    
    return frame

def main():
    cam_mat, dist_coef = load_calibration_data(calib_data_path)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Start video stream from Tello drone
    tello.streamon()
    tello.set_video_direction(1)

    # takeoff
    # tello.takeoff()

    while True:
        time.sleep(SAMPLE_TIME)
        frame = process_frame(tello, FRAME_SIZE)

        # Detect ArUco markers
        corners, ids, _ = detector.detectMarkers(frame, cam_mat, dist_coef)

        # Convert degrees to radians
        roll = math.radians(tello.get_roll())
        pitch = math.radians(tello.get_pitch())

        # Distance height between the drone and the ArUco marker
        height = int(tello.get_distance_tof() * math.cos(roll) * math.cos(pitch)) 

        # If any markers are detected
        if ids is not None:
            # Estimate pose of each marker
            try:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, cam_mat, dist_coef)
            except Exception as e:
                print(f"Error estimating marker pose: {e}")

            control_markers(frame, corners, ids, rvecs, tvecs, height, roll, pitch)
        else:
            # Handle no markers detected
            if height < 150:
                tello.send_rc_control(0, 0, 10, 0)
            else:
                tello.land()

        cv2.imshow("frame", frame)

        # If 'q' is pressed, exit the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    tello.streamoff()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
