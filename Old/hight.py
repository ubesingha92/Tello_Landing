from djitellopy import Tello
from plotter import RealTimePlotter

# Create a Tello object
tello = Tello()

integral = 0
last_error = 0
derivative = 0

ERROR_THRESHOLD = 5  # cm
KP = 3 # Proportional gain
KI = 0.001 # Integral gain
KD = .5   # Derivative gain

def maintain_altitude(target_altitude):
    

    plotter = RealTimePlotter(value_range=(0, 150), title="current altitude")  # Set the value range in the constructor

    try:
        current_altitude = tello.get_distance_tof()

        error = target_altitude - current_altitude
        integral += error
        derivative = error - last_error
        adjustment = int(KP * error + KI * integral + KD * derivative)
        adjustment = min(max(adjustment, -25), 25)

        last_error = error

        # tello.send_rc_control(0, 0, adjustment, 0)

        plotter.update(current_altitude)  # Only pass the current altitude to the update method

    except KeyboardInterrupt:
        print("Exiting...")
        tello.land()
        plotter.finish()

tello.connect()
tello.takeoff()
while True:
    maintain_altitude(50) # replace with your desired altitude
