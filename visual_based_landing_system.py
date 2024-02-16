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
MARKER_SIZE = 5  # cm

target_position = (0, 0)  #  (target_x, target_y) (0,1.5)

# PID coefficients for X 
KP_X, KI_X, KD_X = 0, 0, 0
MAX_SPEED_X = 7.5
# Use target_position[0] as the setpoint for X
pid_x = PID(KP_X, KI_X, KD_X, setpoint=0)
pid_x.output_limits = (-MAX_SPEED_X, MAX_SPEED_X)

# PID coefficients for Y
KP_Y, KI_Y, KD_Y = 0, 0, 0
MAX_SPEED_Y = 7.5
# Use target_position[1] as the setpoint for Y
pid_y = PID(KP_Y, KI_Y, KD_Y, setpoint=0)
pid_y.output_limits = (-MAX_SPEED_Y, MAX_SPEED_Y)

# PID coefficients for yaw
KP_YAW, KI_YAW, KD_YAW = 5, 0.02, .4
MAX_SPEED_YAW = 25
pid_yaw = PID(KP_YAW, KI_YAW, KD_YAW, setpoint=0)
pid_yaw.output_limits = (-MAX_SPEED_YAW, MAX_SPEED_YAW)

XY_ACCEPT_RANGE = 6
XY_ACCEPT_RANGE_LAND = 3
YAW_ACCEPT_RANGE = 10
YAW_ACCEPT_RANGE_LAND = 5
COUNTER_LIMIT_LAND = 2
land_count = 0

MIN_CONTROL_THRESHOLD = 5

# Smoothing factor, adjust as necessary
SMOOTH_FACTOR = 0.1

# Initialize previous filtered values for X and Y
prev_filtered_x = 0
prev_filtered_y = 0

# Connect to Tello drone
tello = Tello()
tello.connect()

filename_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
csv_filename = f'csv/aruco_marker_errors_{filename_time}.csv'

# Initialize control mode
control_mode = "manual"  # Start with manual control

def apply_control_threshold(value):
    if value == 0:
        return 0
    elif value < 0:
        return value - MIN_CONTROL_THRESHOLD
    else:
        return value + MIN_CONTROL_THRESHOLD

def land_drone_safely(tello):
    # Stop all movements
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(.1)
    # Command to land
    tello.land()
    print("Drone is landing safely.")


def load_calibration_data(path):
    # Load calibration data
    calib_data = np.load(path)

    cam_mat = calib_data["camMatrix"]
    dist_coef = calib_data["distCoef"]

    return cam_mat, dist_coef

def write_error_to_csv(file_path, error_data):
    # Column titles
    column_titles = ['Time', 'Marker ID', 'Error X', 'filtered_x', 'Error Y', 'filtered_y', 'Yaw Error', 'Height', 'PID Value X', 'PID Value Y', 'PID Value Yaw', 'Value Z']
    
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

def low_pass_filter_XY(error_x, error_y, prev_filtered_x, prev_filtered_y, SMOOTH_FACTOR):
    filtered_x = round(SMOOTH_FACTOR * error_x + (1 - SMOOTH_FACTOR) * prev_filtered_x, 2)
    filtered_y = round(SMOOTH_FACTOR * error_y + (1 - SMOOTH_FACTOR) * prev_filtered_y, 2)

    # Update previous filtered values
    prev_filtered_x = filtered_x
    prev_filtered_y = filtered_y

    return filtered_x, filtered_y, prev_filtered_x, prev_filtered_y

def draw_frame_center_lines(frame):
    # Calculate the center of the frame
    f_height, f_width, _ = frame.shape
    center_x, center_y = f_width // 2, f_height // 2

    # Draw a vertical line for x=0
    cv2.line(frame, (center_x, 0), (center_x, f_height), (255, 0, 0), 2)

    # Draw a horizontal line for y=0
    cv2.line(frame, (0, center_y), (f_width, center_y), (0, 255, 0), 2)



def control_markers(frame, corners, ids, rvecs, tvecs, height, roll, pitch):
    global land_count, prev_filtered_x, prev_filtered_y

    for i in range(len(ids)):
        cv2.aruco.drawDetectedMarkers(frame, corners)

        # Draw center lines on the frame
        draw_frame_center_lines(frame)

        # Calculate Euler angles
        rotation_matrix, _ = cv2.Rodrigues(rvecs[i])
        euler_angles = cv2.decomposeProjectionMatrix(np.hstack((rotation_matrix, tvecs[i].reshape(3, 1))))[6]

        # Round angles and convert to integer (Roll, Pitch, Yaw)
        euler_angles = [int(round(angle[0])) for angle in euler_angles]
        error_roll, error_pitch, error_yaw = euler_angles

        x, y, z = [(round(item, 2)) for sublist in tvecs[i] for item in sublist]  # Unpack tvecs[i] with two decimal places

        """
        error_x = -x + height * math.tan(roll)
        error_y = y - height * math.tan(pitch)
        """
        error_x = -x + target_position[0]
        error_y = y + target_position[1]

        error_z = z

        pid_val_yaw = -int(pid_yaw(error_yaw))

        if -YAW_ACCEPT_RANGE > error_yaw or error_yaw > YAW_ACCEPT_RANGE:
            print ("--Yaw Correction Only")
            filtered_x, filtered_y, prev_filtered_x, prev_filtered_y = low_pass_filter_XY(0, 0, prev_filtered_x, prev_filtered_y, SMOOTH_FACTOR)
            # PID Control
            pid_val_x = apply_control_threshold(int(pid_x(filtered_x)))
            pid_val_y = apply_control_threshold(int(pid_y(filtered_y)))
            land_count = 0
            val_z = 0

        else:
            filtered_x, filtered_y, prev_filtered_x, prev_filtered_y = low_pass_filter_XY(error_x, error_y, prev_filtered_x, prev_filtered_y, SMOOTH_FACTOR)

            # PID Control
            pid_val_x = apply_control_threshold(int(pid_x(filtered_x)))
            pid_val_y = apply_control_threshold(int(pid_y(filtered_y)))

            # Tello movement control based on PID adjustments
            if (-XY_ACCEPT_RANGE > filtered_x or filtered_x > XY_ACCEPT_RANGE) or (-XY_ACCEPT_RANGE > filtered_y or filtered_y > XY_ACCEPT_RANGE):
                print ("--XY and Yaw Correction")
                land_count = 0
                val_z = 0
                
            else:
                print ("--Ok")
                val_z = -15
                if (height <= 32 and (-XY_ACCEPT_RANGE_LAND < filtered_x < XY_ACCEPT_RANGE_LAND) and (-XY_ACCEPT_RANGE_LAND < filtered_y < XY_ACCEPT_RANGE_LAND) and (-YAW_ACCEPT_RANGE_LAND < error_yaw < YAW_ACCEPT_RANGE_LAND)):
                    if(land_count > COUNTER_LIMIT_LAND):
                        land_drone_safely(tello)
                        return 1
                    else:
                        filtered_x, filtered_y, prev_filtered_x, prev_filtered_y = low_pass_filter_XY(0, 0, prev_filtered_x, prev_filtered_y, SMOOTH_FACTOR)
                        land_count += 1
        
        print (f" X: {error_x} F_X:  {filtered_x}  Y: {error_y} F_Y:  {filtered_y} Yaw: {error_yaw} Height: {height}", end="")

        # Write errors to CSV
        human_readable_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        error_data = [human_readable_time, ids[i][0], error_x, filtered_x, error_y, filtered_y, error_yaw, height, pid_val_x, pid_val_y, pid_val_yaw, val_z]
        write_error_to_csv(csv_filename, error_data)

        # RC Control
        tello.send_rc_control(int(pid_val_x), int(pid_val_y), val_z, int(pid_val_yaw))

        return 0

def process_frame(tello, frame_size):
    # Read frame from the drone
    frame = tello.get_frame_read().frame

    # Rotate the frame and crop around the center
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = frame[frame.shape[0] - frame_size[0]:frame.shape[0],
            frame.shape[1] - frame_size[1]:frame.shape[1]]
    
    return frame

def switch_control_mode():
    global control_mode
    if control_mode == "manual":
        control_mode = "auto"
    else:
        control_mode = "manual"

def handle_manual_control(tello, key):
    speed = 30  # Adjust speed as needed
    if key == ord('w'):  # Forward
        tello.send_rc_control(0, speed, 0, 0)
    elif key == ord('s'):  # Backward
        tello.send_rc_control(0, -speed, 0, 0)
    elif key == ord('a'):  # Left
        tello.send_rc_control(-speed, 0, 0, 0)
    elif key == ord('d'):  # Right
        tello.send_rc_control(speed, 0, 0, 0)
    elif key == ord('q'):  # Rotate left
        tello.send_rc_control(0, 0, 0, -speed)
    elif key == ord('e'):  # Rotate right
        tello.send_rc_control(0, 0, 0, speed)
    elif key == ord('r'):  # Up
        tello.send_rc_control(0, 0, speed, 0)
    elif key == ord('f'):  # Down
        tello.send_rc_control(0, 0, -speed, 0)
    elif key == ord('t'):  # Takeoff
        tello.takeoff()
    else:  # Stop
        tello.send_rc_control(0, 0, 0, 0)

def display_frame_with_text(frame, text):
    """
    Adds specified text to the top left corner of the given frame and displays it.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (10, 50)
    fontScale = 1
    color = (0, 255, 0)  # Green color
    thickness = 2

    cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("Tello Video Stream", frame)

def main():
    global prev_filtered_x, prev_filtered_y
    cam_mat, dist_coef = load_calibration_data(calib_data_path)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Start video stream from Tello drone
    tello.streamon()
    tello.set_video_direction(1)

    # takeoff
    # tello.takeoff()
    tello.send_rc_control(0, 0, 0, 0)

    while True:
        time.sleep(.01)
        frame = process_frame(tello, FRAME_SIZE)

        # Detect ArUco markers
        corners, ids, _ = detector.detectMarkers(frame, cam_mat, dist_coef)

        # Convert degrees to radians
        roll = math.radians(tello.get_roll())
        pitch = math.radians(tello.get_pitch())

        # Distance height between the drone and the ArUco marker
        height = int(tello.get_distance_tof() * math.cos(roll) * math.cos(pitch))

        # Get Battery
        battery_value = tello.get_battery()
        print (f"Battery: {battery_value}", end="")

        if(battery_value<20):
            land_drone_safely(tello)
            break

        frame = process_frame(tello, FRAME_SIZE)  # Capture the current frame

        # Check for control mode switching and other commands
        key = cv2.waitKey(10) & 0xFF
        if key == ord(' '):  # Toggle control mode on space press
            switch_control_mode()
        elif key == ord('l'):  # Quit and land on 'l' press
            land_drone_safely(tello)
            break

        if control_mode == "manual":
            display_frame_with_text(frame, "M")
            handle_manual_control(tello, key)
            continue  # Skip the rest of the loop to avoid automatic control processing

        # If any markers are detected
        if ids is not None:
            # Estimate pose of each marker
            try:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, cam_mat, dist_coef)
            except Exception as e:
                print(f"Error estimating marker pose: {e}")

            if(control_markers(frame, corners, ids, rvecs, tvecs, height, roll, pitch) == 1):
                break
        else:
            # Handle no markers detected
            print ("No Markers")
            _, _, prev_filtered_x, prev_filtered_y = low_pass_filter_XY(0, 0, prev_filtered_x, prev_filtered_y, SMOOTH_FACTOR)
            val_z = 0
            
            if height < 150:
                val_z = 0
            else:
                val_z = 0

            tello.send_rc_control(0, 0, val_z, 0)
            # Write errors to CSV
            human_readable_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            error_data = [human_readable_time, None, None, None, None, None, None, height, None, None, None, val_z]
            write_error_to_csv(csv_filename, error_data)

        display_frame_with_text(frame, "A")

        # If 'q' is pressed, exit the loop
        if cv2.waitKey(10) & 0xFF == ord('l'):
            land_drone_safely(tello)
            break

    tello.streamoff()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
