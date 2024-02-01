import cv2
import numpy as np
from djitellopy import Tello
import math
import time

# Define frame size
FRAME_SIZE = (320, 240)

# Path to calibration data
calib_data_path = "Camera_Calibartion/calib.npz"

# Size of the ArUco markers (in centimeters)
MARKER_SIZE = 7.5  # cm

# PID coefficients for X
KP_X, KI_X, KD_X = 3, 0, 1.5
MAX_SPEED_X = 15

# Initialize global variables for X
x_integral = 0
x_last_error = 0
x_derivative = 0

# PID coefficients for Y
KP_Y, KI_Y, KD_Y = 3, 0, 1.5
MAX_SPEED_Y = 15

# Initialize global variables for Y
y_integral = 0
y_last_error = 0
y_derivative = 0

# PID coefficients for Z
KP_Z, KI_Z, KD_Z = 0, 0, 0
MAX_SPEED_Z = 20

# Initialize global variables for Z
z_integral = 0
z_last_error = 0
z_derivative = 0

# PID coefficients for yaw
KP_YAW, KI_YAW, KD_YAW = 4, 0.01, 1
MAX_SPEED_YAW = 40

XY_ACCEPT_RANGE = 10
YAW_ACCEPT_RANGE = 10

ok_counter = 0
OK_COUNTER_LIMIT = 1

# Initialize global variables
yaw_integral = 0
yaw_last_error = 0
yaw_derivative = 0

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
    # Global variables for XYZ and yaw PID controller
    global ok_counter
    global x_integral, x_last_error, x_derivative
    global y_integral, y_last_error, y_derivative
    global z_integral, z_last_error, z_derivative
    global yaw_integral, yaw_last_error, yaw_derivative

    # Convert degrees to radians
    roll = math.radians(tello.get_roll())
    pitch = math.radians(tello.get_pitch())

    # distance height between the drone and the ArUco marker
    height = tello.get_distance_tof() * math.cos(roll) * math.cos(pitch)

    for i in range(len(ids)):
        cv2.aruco.drawDetectedMarkers(frame, corners)

        x, y, _ = [int(round(item)) for sublist in tvecs[i] for item in sublist]  # Unpack and round tvecs[i]

        # correction
        x = -x + height * math.sin(roll)
        y = y - height * math.sin(pitch)

        # Calculate Euler angles
        rotation_matrix, _ = cv2.Rodrigues(rvecs[i])
        euler_angles = cv2.decomposeProjectionMatrix(np.hstack((rotation_matrix, tvecs[i].reshape(3, 1))))[6]

        # Round angles and convert to integer (Roll, Pitch, Yaw)
        euler_angles = [int(round(angle[0])) for angle in euler_angles]
        yaw_value = euler_angles[2]

        # get Battery
        battery_value = tello.get_battery()

        # Print distance and angles to terminal
        # print(f"Battery: {battery_value} ID: {ids[i][0]} X: {round(x)}cm Y: {round(y)}cm Height: {round(height)}cm Yaw Value: {round(yaw_value)} Roll: {round(math.degrees(roll))} Pitch: {round(math.degrees(pitch))}")

        # X-axis PID controller
        x_error = x
        x_integral += x_error
        x_derivative = x_error - x_last_error

        x_adjustment = int(KP_X * x_error + KI_X * x_integral + KD_X * x_derivative)
        x_adjustment = -min(max(x_adjustment, -MAX_SPEED_X), MAX_SPEED_X)

        x_last_error = x_error

        # Y-axis PID controller
        y_error = y
        y_integral += y_error
        y_derivative = y_error - y_last_error

        y_adjustment = int(KP_Y * y_error + KI_Y * y_integral + KD_Y * y_derivative)
        y_adjustment = -min(max(y_adjustment, -MAX_SPEED_Y), MAX_SPEED_Y)

        y_last_error = y_error

        # Z-axis PID controller
        z_error = height
        z_integral += z_error
        z_derivative = z_error - z_last_error

        z_adjustment = int(KP_Z * z_error + KI_Z * z_integral + KD_Z * z_derivative)
        z_adjustment = min(max(z_adjustment, -MAX_SPEED_Z), MAX_SPEED_Z)

        z_last_error = z_error
        
        # yaw_value
        yaw_error = yaw_value
        yaw_integral += yaw_error
        yaw_derivative = yaw_error - yaw_last_error

        yaw_adjustment = int(KP_YAW * yaw_error + KI_YAW * yaw_integral + KD_YAW * yaw_derivative)
        yaw_adjustment = min(max(yaw_adjustment, -MAX_SPEED_YAW), MAX_SPEED_YAW)

        yaw_last_error = yaw_error

        print (f"Battery: {battery_value} X: {x_adjustment} yaw: {yaw_adjustment}", end="")

        # tello.send_rc_control(-x_adjustment, y_adjustment, z_adjustment, yaw_adjustment)
        if -YAW_ACCEPT_RANGE > yaw_adjustment or  yaw_adjustment > YAW_ACCEPT_RANGE:
            tello.send_rc_control(0, 0, 0, yaw_adjustment)
            print ("--Need Yaw Correction")
            ok_counter = 0
        elif (-XY_ACCEPT_RANGE > x_adjustment or  x_adjustment > XY_ACCEPT_RANGE) and (-XY_ACCEPT_RANGE > y_adjustment or  y_adjustment > XY_ACCEPT_RANGE):
            tello.send_rc_control(x_adjustment, y_adjustment, 0, yaw_adjustment)
            print ("--Need XY Correction")
            ok_counter = 0
        else:
            print ("--Ok")
            tello.send_rc_control(x_adjustment, y_adjustment, 0, yaw_adjustment)
            ok_counter += 1
            if ok_counter >= OK_COUNTER_LIMIT:
                tello.send_rc_control(x_adjustment, y_adjustment, -20, yaw_adjustment)
                if( height<= 35):
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

    # # takeoff
    tello.takeoff()
    tello.move_down(30)

    while True:
        time.sleep(.2)
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
             tello.send_rc_control(0, 0, 0, 0)

        cv2.imshow("frame", frame)

        # If 'q' is pressed, exit the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    tello.streamoff()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
