import cv2
import numpy as np
from djitellopy import Tello
import math


# Define frame size 
FRAME_SIZE = (320, 240)

# Path to calibration data
calib_data_path = "Camera_Calibartion/calib.npz"

# Size of the ArUco markers (in centimeters)
MARKER_SIZE = 5

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
    
    # Convert degrees to radians
    roll = math.radians(tello.get_roll())
    pitch = math.radians(tello.get_pitch())
   
    # distance height between the drone and the ArUco marker
    height = tello.get_distance_tof() * math.cos(roll) * math.cos(pitch)

    for i in range(len(ids)):
        cv2.aruco.drawDetectedMarkers(frame, corners)

        x, y, _ = [int(round(item)) for sublist in tvecs[i] for item in sublist]  # Unpack and round tvecs[i]

        # corection
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
        print(f"Battery: {battery_value} ID: {ids[i][0]} X: {round(x)}cm Y: {round(y)}cm Height: {round(height)}cm Yaw Value: {round(yaw_value)} Roll: {round(math.degrees(roll))} Pitch: {round(math.degrees(pitch))}")

def process_frame(tello, frame_size):
    # Read frame from the drone
    frame = tello.get_frame_read().frame  

    # Rotate the frame and crop around the center
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = frame[frame.shape[0] - frame_size[0] : frame.shape[0], 
                  frame.shape[1] - frame_size[1] : frame.shape[1]]

    return frame

def main():
    cam_mat, dist_coef = load_calibration_data(calib_data_path)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Start video stream from Tello drone
    tello.streamon()
    tello.set_video_direction(1)

    while True:
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

        cv2.imshow("frame", frame)

        # If 'q' is pressed, exit the loop
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

    tello.streamoff()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
