from djitellopy import Tello
import time

def test_drone_connection():
    # Create a Tello object
    drone = Tello()

    try:
        # Connect to the drone
        drone.connect()
        print("Connected to Tello Drone")

        # Get the battery level
        battery_level = drone.get_battery()
        print(f'Battery Level: {battery_level}%')

        # Take off
        print("Taking off...")
        drone.takeoff()
        time.sleep(5)  # Give time for the drone to take off

        # Perform any additional commands or maneuvers here
        # ...

        # Land
        print("Landing...")
        drone.land()

    except Exception as e:
        print(f'Error: {e}')
    finally:
        # Always safely disconnect
        drone.end()

if __name__ == '__main__':
    test_drone_connection()
