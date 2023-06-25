import math
from djitellopy import Tello
import time

# Initialize Tello drone
drone = Tello()

# Connect to the drone
drone.connect()

while True:
    # Get the roll and pitch angles in degrees
    roll = drone.get_roll()
    pitch = drone.get_pitch()

    print("Roll: ", roll, "   Pitch: ", pitch)
    time.sleep(.2)

# Disconnect from the drone
drone.disconnect()