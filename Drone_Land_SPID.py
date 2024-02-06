import cv2
import numpy as np
from djitellopy import Tello
import math
import time
from simple_pid import PID

# Define frame size
FRAME_SIZE = (320, 240)

# Path to calibration data
calib_data_path = "Camera_Calibartion/calib.npz"

# Size of the ArUco markers (in centimeters)
MARKER_SIZE = 7.5  # cm

# PID coefficients for X
# KP_X, KI_X, KD_X = 3, 0, 5
KP_X, KI_X, KD_X = 0, 0, 0
MAX_SPEED_X = 15
pid_x = PID(KP_X, KI_X, KD_X, setpoint=0)
pid_x.output_limits = (-MAX_SPEED_X, MAX_SPEED_X)

# PID coefficients for Y
# KP_Y, KI_Y, KD_Y = 3, 0, 5
KP_Y, KI_Y, KD_Y = 0, 0, 0
MAX_SPEED_Y = 15
pid_y = PID(KP_Y, KI_Y, KD_Y, setpoint=0)
pid_y.output_limits = (-MAX_SPEED_Y, MAX_SPEED_Y)

# PID coefficients for yaw
# KP_YAW, KI_YAW, KD_YAW = 4, 0.01, 1
KP_YAW, KI_YAW, KD_YAW = 4, 0.01, 1
MAX_SPEED_YAW = 15
pid_yaw = PID(KP_YAW, KI_YAW, KD_YAW, setpoint=0)
pid_yaw.output_limits = (-MAX_SPEED_YAW, MAX_SPEED_YAW)

XY_ACCEPT_RANGE = 3
YAW_ACCEPT_RANGE = 3
OK_COUNTER_LIMIT = 1

# Connect to Tello drone
tello = Tello()
tello.connect()

def load_calibration_data(path):
    # Load calibration data
    calib_data = np.load(path)

    cam_mat = calib_data["camMatrix"]
    dist_coef = calib_data["distCoef"]

    return cam_mat, dist_coef

def draw_detected_markers(frame, corners, ids, rvecs, tvecs):
    global ok_counter

    # Convert degrees to radians
    roll = math.radians(tello.get_roll())
    pitch = math.radians(tello.get_pitch())

    # Distance height between the drone and the ArUco marker
    height = tello.get_distance_tof() * math.cos(roll) * math.cos(pitch)

    for i in range(len(ids)):
        cv2.aruco.drawDetectedMarkers(frame, corners)

        x, y, _ = [int(round(item)) for sublist in tvecs[i] for item in sublist]  # Unpack and round tvecs[i]

        # Correction
        x = -x + height * math.sin(roll)
        y = y - height * math.sin(pitch)

        # Calculate Euler angles
        rotation_matrix, _ = cv2.Rodrigues(rvecs[i])
        euler_angles = cv2.decomposeProjectionMatrix(np.hstack((rotation_matrix, tvecs[i].reshape(3, 1))))[6]

        # Round angles and convert to integer (Roll, Pitch, Yaw)
        euler_angles = [int(round(angle[0])) for angle in euler_angles]
        yaw_value = euler_angles[2]

        # Get Battery
        battery_value = tello.get_battery()

        # PID Control
        x_adjustment = int(pid_x(x))
        y_adjustment = int(pid_y(y))
        yaw_adjustment = int(pid_yaw(yaw_value))

        print (f"Battery: {battery_value} X: {x_adjustment} Y: {y_adjustment} Yaw: {yaw_adjustment}", end="")

        # Tello movement control based on PID adjustments
        if -YAW_ACCEPT_RANGE > yaw_value or yaw_value > YAW_ACCEPT_RANGE:
            tello.send_rc_control(0, 0, 0, yaw_adjustment)
            print ("--Need Yaw Correction")
            ok_counter = 0
        elif (-XY_ACCEPT_RANGE > x or x > XY_ACCEPT_RANGE) and (-XY_ACCEPT_RANGE > y or y > XY_ACCEPT_RANGE):
            tello.send_rc_control(x_adjustment, y_adjustment, 0, yaw_adjustment)
            print ("--Need XY Correction")
            ok_counter = 0
        else:
            print ("--Ok")
            tello.send_rc_control(x_adjustment, y_adjustment, 0, yaw_adjustment)
            ok_counter += 1
            if ok_counter >= OK_COUNTER_LIMIT:
                tello.send_rc_control(x_adjustment, y_adjustment, -15, yaw_adjustment)
                if height <= 35:
                    tello.land()

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
    tello.takeoff()

    while True:
        time.sleep(0.2)
        frame = process_frame(tello, FRAME_SIZE)

        # Detect ArUco markers
        corners, ids, _ = detector.detectMarkers(frame)

        # If any markers are detected
        if ids is not None:
            # Estimate pose of each marker
            try:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, cam_mat, dist_coef)
            except Exception as e:
                print(f"Error estimating marker pose: {e}")

            draw_detected_markers(frame, corners, ids, rvecs, tvecs)
        else:
            # Handle no markers detected
            roll = math.radians(tello.get_roll())
            pitch = math.radians(tello.get_pitch())
            height = tello.get_distance_tof() * math.cos(roll) * math.cos(pitch)
            if height < 100:
                tello.send_rc_control(0, 0, 10, 0)
            else:
                tello.send_rc_control(0, 0, 0, 0)

        cv2.imshow("frame", frame)

        # If 'q' is pressed, exit the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    tello.streamoff()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
