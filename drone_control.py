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
# PID coefficients for X and Y
KP_XY, KI_XY, KD_XY = .3, 0.00001, .1
MAX_SPEED_XY = 25

# PID coefficients for Altitude
KP_H, KI_H, KD_H = 3, 0.001, 1

# Calculate the center of the frame
FRAME_CENTER_X = 120
FRAME_CENTER_Y = 160

MARKER_SIZE = 5  # 5cm

# Initialize global variables
left_right_velocity = 0
forward_backward_velocity = 0
up_down_velocity = 0
yaw_velocity = 0

current_altitude = 0
battery_level = 0

x_integral = 0
x_last_error = 0
x_derivative = 0

y_integral = 0
y_last_error = 0
y_derivative = 0

h_integral = 0
h_last_error = 0
h_derivative = 0


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


def process_frame():
    frame = tello.get_frame_read().frame
    
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # If the frame size exceeds twice the center coordinates, return None
    frame = frame[frame.shape[0]-2*FRAME_CENTER_Y:frame.shape[0], frame.shape[1]-2*FRAME_CENTER_X:frame.shape[1]]

    cv2.imshow("drone", frame)
    cv2.waitKey(1)
    return frame


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


def maintain_x_position(current_position_x, marker_size_in_image):
    global x_integral, x_last_error, x_derivative

    if current_position_x is None:
        # tello.send_rc_control(0, 0, 0, 0)
        return None
    
    # assuming that the MARKER_SIZE is in cm and marker_size_in_image is in pixels
    pixel_to_size = MARKER_SIZE / marker_size_in_image

    x_error = (current_position_x - FRAME_CENTER_X) * pixel_to_size
    x_integral += x_error
    x_derivative = x_error - x_last_error

    x_adjustment = int(KP_XY * x_error + KI_XY * x_integral + KD_XY * x_derivative)
    x_adjustment = min(max(x_adjustment, -MAX_SPEED_XY), MAX_SPEED_XY)

    print(x_error)

    # tello.send_rc_control(x_adjustment, 0, 0, 0)

    x_last_error = x_error

    return x_error


def maintain_y_position(current_position_y, marker_size_in_image):
    global y_integral, y_last_error, y_derivative

    if current_position_y is None:
        # tello.send_rc_control(0, 0, 0, 0)
        return None

    # assuming that the MARKER_SIZE is in cm and marker_size_in_image is in pixels
    pixel_to_size = MARKER_SIZE / marker_size_in_image

    # calculate y_error in cm
    y_error = (FRAME_CENTER_Y - current_position_y) * pixel_to_size

    y_integral += y_error
    y_derivative = y_error - y_last_error

    y_adjustment = int(KP_XY * y_error + KI_XY * y_integral + KD_XY * y_derivative)
    y_adjustment = min(max(y_adjustment, -MAX_SPEED_XY), MAX_SPEED_XY)

    print(y_error)

    # tello.send_rc_control(0, y_adjustment, 0, 0)

    y_last_error = y_error

    return y_error



def maintain_altitude(target_altitude):
    global h_integral, h_last_error, h_derivative, current_altitude

    current_altitude = tello.get_distance_tof()

    h_error = target_altitude - current_altitude
    h_integral += h_error
    h_derivative = h_error - h_last_error

    h_adjustment = int(KP_H * h_error + KI_H * h_integral + KD_H * h_derivative)
    h_adjustment = min(max(h_adjustment, -25), 25)

    tello.send_rc_control(0, 0, h_adjustment, 0)

    h_last_error = h_error

    return h_error



def downward_detector_aruco(ARUCO_ID):
    for i in range(100):
        frame = process_frame()

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters =  cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None and ARUCO_ID in ids:
            print("ArUco marker ", ARUCO_ID, " detected")
            index = list(ids.flatten()).index(ARUCO_ID)
            marker_corners = corners[index]

            # Calculate the perimeter of the ArUco marker
            marker_perimeter = cv2.arcLength(marker_corners[0], True)

            # Calculate the center of the ArUco marker
            center_x = int(np.mean([marker_corners[0][i][0] for i in range(4)]))
            center_y = int(np.mean([marker_corners[0][i][1] for i in range(4)]))

            # Draw rectangle around the detected ArUco marker
            cv2.aruco.drawDetectedMarkers(frame, [marker_corners])

            # Draw a red dot on the center of the ArUco marker
            cv2.circle(frame, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)

            # Draw a line from the center of the frame to the blue line
            cv2.line(frame, (FRAME_CENTER_X, FRAME_CENTER_Y), (center_x, center_y), color=(255, 0, 0), thickness=2)

            # Show the frame with the drawn dot and line
            cv2.imshow("drone", frame)
            cv2.waitKey(1)
            
            # break
            return center_x, center_y, marker_perimeter

    return None, None, None



def finder_aruco():
    # This function will continuously move the drone and check its video feed to find an ArUco marker.
     while True:
        marker = downward_detector_aruco(ARUCO_ID)

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
    
    # tello.turn_motor_on()


    # listener = Listener(on_press=on_press, on_release=on_release)
    # listener.start()


    global plotter
    plotter = RealTimePlotter(title="Drone Controlling")  # Set the value range in the constructor

    # tello.set_video_direction(1) # Set the Tello drone's video direction to downward facing
    # frame = tello.get_frame_read().frame
    # cv2.imshow("drone", frame)
    process_frame()
    # tello.takeoff()
    video_start = tello.set_video_direction(1)
    # time.sleep(1)
    
    # frame = tello.get_frame_read().frame
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # frame = cv2.flip(frame, 0)  # flip along x-axis
    # cv2.imshow("drone", frame)
    # cv2.waitKey(1)


    try:
        # while True:
        #     if not handle_low_battery():
        #         break
        #     battery_level = tello.get_battery()
        #     h_error = maintain_altitude(50)
        #     plotter.update(0, 0, h_error)
        #     if abs(h_error) < ERROR_THRESHOLD:
        #         # tello.send_rc_control(0, 0, 0, 0)
        #         print("50")
        #         time.sleep(5)
        #         break
        while True:
            print("battery: ", tello.get_battery(), "%")
            x_position, y_position, marker_perimeter = downward_detector_aruco(20)

            # x_error = maintain_x_position(x_position)
            y_error = maintain_y_position(y_position, marker_perimeter)

            plotter.update(0, y_error, 0)

        while True:
            while True:
                if not handle_low_battery():
                    break
                battery_level = tello.get_battery()
                h_error = maintain_altitude(50)
                plotter.update(0, 0, h_error, battery_level)
                if abs(h_error) < ERROR_THRESHOLD:
                    print("50")
                    break
                    
                
            while True:
                if not handle_low_battery():
                    break
                battery_level = tello.get_battery()
                h_error = maintain_altitude(80)
                plotter.update(0, 0, h_error, battery_level)
                if abs(h_error) < ERROR_THRESHOLD:
                    print("80")
                    break
                

    except KeyboardInterrupt:
        print("Exiting...")
        tello.end()
        plotter.finish()    



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        tello.end()
        plotter.finish()  
    except Exception as e:
        print(f"An error occurred: {e}")
        tello.end()
        plotter.finish()  