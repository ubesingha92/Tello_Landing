# Visual Based Landing System for Tello Drone

This repository contains a Python script designed for controlling a Tello drone using visual markers (specifically, ArUco markers) for precise navigation and landing. The system utilizes the drone's camera to detect markers, calculate position and orientation errors, and adjust the drone's flight path accordingly using PID control. This approach allows for precise maneuvers and safe landings based on visual cues.

## Features

- **Camera Calibration**: Incorporates camera calibration data to correct lens distortion for accurate marker detection.
- **PID Control**: Utilizes Proportional-Integral-Derivative (PID) controllers to manage the drone's x, y, and yaw movements for smooth navigation.
- **Safety Measures**: Implements safety mechanisms for low battery situations and manual override, ensuring safe landing.
- **Error Logging**: Records navigation errors and control values for analysis, aiding in system tuning and debugging.
- **Dynamic Adjustment**: Adjusts the drone's movements in real-time based on detected marker positions, enabling precise positioning and landing.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- A Tello drone.
- Python 3.6 or higher installed on your system.
- The following Python libraries installed:
  - `djitellopy` for controlling the Tello drone.
  - `opencv-python` for image processing and marker detection.
  - `numpy` for numerical operations.
  - `simple_pid` for implementing PID control.

## Installation

To use the Visual Based Landing System, follow these steps:

1. Clone the repository to your local machine.
2. Install the required Python libraries:
   ```
   pip install djitellopy opencv-python numpy simple-pid
   ```
3. Ensure your Tello drone's firmware is up to date and that it is connected to your computer's Wi-Fi network.

## Usage

1. Place ArUco markers on the ground where you want the drone to navigate and land. Ensure the markers are visible and unobstructed.
2. Load the camera calibration data into the script. If you haven't calibrated your camera, follow OpenCV's camera calibration tutorials to generate `calib.npz`.
3. Run the script:
   ```
   python visual_based_landing_system.py
   ```
4. The script will connect to the drone, start the video stream, and begin detecting ArUco markers. The drone will adjust its position and orientation based on the detected markers and perform a safe landing when positioned correctly above a marker.

### Important Note on Firewall Settings

To ensure successful communication between your computer and the Tello drone, you may need to temporarily disable your firewall or configure specific rules to allow traffic from the drone. However, proceed with caution, as disabling the firewall may expose your system to security risks. Always re-enable the firewall after testing is complete. If unsure, consult with an IT professional or refer to your firewall's documentation on how to safely configure exceptions.

## Customization

You can adjust the following parameters in the script based on your setup and requirements:
- `FRAME_SIZE`: The size of the video frame from the drone's camera.
- `calib_data_path`: The path to your camera calibration data file.
- `MARKER_SIZE`: The size of your ArUco markers in centimeters.
- PID coefficients for x, y, and yaw movements (`KP_X`, `KI_X`, `KD_X`, etc.).
- Various thresholds for movement control and landing.

## Contributing

Contributions to improve the Visual Based Landing System are welcome. Feel free to fork the repository and submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The Tello drone team for providing an accessible and versatile drone platform.
- The OpenCV team for the powerful image processing library.
- The `djitellopy`, `numpy`, and `simple_pid` library authors for their excellent work.
