import math
from djitellopy import Tello
import time

# Initialize Tello drone
tello = Tello()

# Connect to the drone
tello.connect()

try:
    while True:
        # Get the roll and pitch angles in degrees and convert degrees to radians
        roll = math.radians(tello.get_roll())
        pitch = math.radians(tello.get_pitch())

        # Distance height between the drone and the ArUco marker
        height = round(tello.get_distance_tof() * math.cos(roll) * math.cos(pitch))

        print(f"Roll: {round(math.degrees(roll))} Pitch: {round(math.degrees(pitch))}")

        # print("Height: ", height)
        time.sleep(0.2)

except KeyboardInterrupt:
    print("Keyboard interrupt detected. Stopping the program.")

# Disconnect from the Tello
tello.disconnect()
