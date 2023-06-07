# cd D:\MPhile\2.2 Land\Landing>
# python -m venv myenv
# .\myenv\Scripts\activate

from djitellopy import Tello
import cv2
import numpy as np
from pynput.keyboard import Key, Listener, KeyCode
import time
from plotter import RealTimePlotter


# Constant
KEYBOARD_SPEED = 30
ARUCO_ID = 20
LOW_BATTERY_THRESHOLD = 30
FINDER_MAX_HIGHT = 70

ERROR_THRESHOLD = 2  # cm
KP = 3 # Proportional gain
KI = 0.001 # Integral gain
KD = 1   # Derivative gain

# Initialize global variables
left_right_velocity = 0
forward_backward_velocity = 0
up_down_velocity = 0
yaw_velocity = 0

integral_h = 0
last_error_h = 0
derivative_h = 0


tello = None
plotter = None


def initialize_drone():
    global tello
    tello = Tello()

    try:
        tello.connect()
    except Exception as e:
        print(f"Failed to connect to drone: {e}")
        return False

    tello.streamon()
    return True


def handle_low_battery():
    battery_level = tello.get_battery()
    if battery_level < LOW_BATTERY_THRESHOLD:
        print("Battery level is low. Landing the drone.")
        tello.end()
        return False
    return True


def rotate_frame(frame):
    # Rotate the frame clockwise by 90 degrees
    rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return rotated_frame


def get_video_display_down():
    tello.set_video_direction(1)
    try:
        cv2.waitKey(1)
        frame = tello.get_frame_read().frame
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow("drone", rotated_frame)
        return rotated_frame
    except KeyboardInterrupt:
        print("Exiting...")
        tello.land()


def on_press(key):
    global left_right_velocity
    global forward_backward_velocity
    global up_down_velocity
    global yaw_velocity

    # Determine action based on key press
    if key == Key.up:
        forward_backward_velocity = KEYBOARD_SPEED
    elif key == Key.down:
        forward_backward_velocity = -KEYBOARD_SPEED
    elif key == Key.left:
        left_right_velocity = -KEYBOARD_SPEED
    elif key == Key.right:
        left_right_velocity = KEYBOARD_SPEED
    elif key == KeyCode.from_char('w'):
        up_down_velocity = KEYBOARD_SPEED
    elif key == KeyCode.from_char('s'):
        up_down_velocity = -KEYBOARD_SPEED
    elif key == KeyCode.from_char('a'):
        yaw_velocity = -KEYBOARD_SPEED
    elif key == KeyCode.from_char('d'):
        yaw_velocity = KEYBOARD_SPEED
    elif key == KeyCode.from_char('t'):
        tello.takeoff()
    elif key == KeyCode.from_char('l'):
        tello.land()

    # Send the drone's velocity commands to the drone
    tello.send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity)


def on_release(key):
    global left_right_velocity
    global forward_backward_velocity
    global up_down_velocity
    global yaw_velocity

    # Reset velocities based on key release
    if key == Key.up or key == Key.down:
        forward_backward_velocity = 0
    elif key == Key.left or key == Key.right:
        left_right_velocity = 0
    elif key == KeyCode.from_char('w') or key == KeyCode.from_char('s'):
        up_down_velocity = 0
    elif key == KeyCode.from_char('a') or key == KeyCode.from_char('d'):
        yaw_velocity = 0

    # Send the drone's velocity commands to the drone
    tello.send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity)


def maintain_altitude(target_altitude):
    global integral_h, last_error_h, derivative_h  # declare global variables
    try:
        current_altitude = tello.get_distance_tof()

        error_h = target_altitude - current_altitude
        integral_h += error_h
        derivative_h = error_h - last_error_h

        adjustment = int(KP * error_h + KI * integral_h + KD * derivative_h)
        adjustment = min(max(adjustment, -25), 25)

        tello.send_rc_control(0, 0, adjustment, 0)

        last_error_h = error_h

        plotter.update(current_altitude)

        return abs(error_h)

    except KeyboardInterrupt:
        print("Exiting...")
        tello.end()
        plotter.finish()


def downward_detecter_aruco(ARUCO_ID):
    for i in range(100):
        frame = tello.get_frame_read().frame
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow("drone", frame)
        cv2.waitKey(1)

        # Load ArUco dictionary, initialize detection parameters
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()

        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        if ids is not None and ARUCO_ID in ids:
            print("ArUco marker ", ARUCO_ID, " detected")
            index = list(ids.flatten()).index(ARUCO_ID)
            marker_corners = corners[index]

            # Calculate the center of the ArUco marker
            center_x = int(np.mean([marker_corners[0][i][0] for i in range(4)]))
            center_y = int(np.mean([marker_corners[0][i][1] for i in range(4)]))

            # Draw rectangle around the detected ArUco marker
            cv2.aruco.drawDetectedMarkers(frame, [marker_corners])

            # Draw a red dot on the center of the ArUco marker
            cv2.circle(frame, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)

            # Calculate the center of the frame
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2

            # Draw a line from the center of the frame to the red dot
            cv2.line(frame, (frame_center_x, frame_center_y), (center_x, center_y), color=(0, 0, 255), thickness=2)

            # Show the frame with the drawn dot and line
            cv2.imshow("drone", frame)
            cv2.waitKey(1)

            return center_x, center_y, corners

    return None


def finder_aruco():
    # This function will continuously move the drone and check its video feed to find an ArUco marker.
    
    while True:
        marker = downward_detecter_aruco(ARUCO_ID)

        if marker is not None:
            break
        else:
            if tello.get_height_() > FINDER_MAX_HIGHT:
                print("Error: Altitude is higher than ",FINDER_MAX_HIGHT)
                """wateta kerakila ballan ethakota lesi wewi
                """
                # # Move the drone in a pattern for different viewpoints.
                # tello.move_forward(50)
                # marker = detecter_aruco(ARUCO_ID)
                # if marker is not None:
                #     break
            else:
                tello.move_up(20)


def main():
    if not initialize_drone():
        print("Failed to initialize drone.")
        return
    
    tello.turn_motor_on()
    # listener = Listener(on_press=on_press, on_release=on_release)
    # listener.start()
    # global plotter
    # plotter = RealTimePlotter(value_range=(0, 150), title="current altitude")  # Set the value range in the constructor
    # tello.set_video_direction(1) # Set the Tello drone's video direction to downward facing
    # frame = tello.get_frame_read().frame
    # cv2.imshow("drone", frame)

    # tello.takeoff()
    tello.set_video_direction(1)
    frame = tello.get_frame_read().frame
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow("drone", frame)
    cv2.waitKey(1)

    try:
        while True:
            while True:
                if not handle_low_battery():
                    break
                # get_video_display_down()
                downward_detecter_aruco(ARUCO_ID)
                # print(tello.get_battery(),"   ", tello.get_distance_tof())
                # tello.set_video_direction(1) # Set the Tello drone's video direction to downward facing
                # if maintain_altitude(50)<ERROR_THRESHOLD:
                    # print("50")
                    # break
                    
                
            # while True:
            #     if not handle_low_battery():
            #         break
            #     # get_video_display_down()
            #     # detecter_downward_aruco(ARUCO_ID)
            #     # print(tello.get_battery(),"   ", tello.get_distance_tof())
            #     # tello.set_video_direction(1) # Set the Tello drone's video direction to downward facing
            #     if maintain_altitude(80)<ERROR_THRESHOLD:
            #         # print("80")
            #         break
                

    except KeyboardInterrupt:
        print("Exiting...")
        tello.end()
        plotter.finish()    



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        tello.end()
    except Exception as e:
        print(f"An error occurred: {e}")
        tello.end()