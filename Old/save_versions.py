import numpy
import cv2
import djitellopy

# Get library versions
numpy_version = numpy.__version__
opencv_version = cv2.__version__
djitellopy_version = djitellopy.__version__

# Create a requirements.txt file and write library versions to it
with open("requirements.txt", "w") as f:
    f.write(f"numpy=={numpy_version}\n")
    f.write(f"opencv-python-headless=={opencv_version}\n")
    f.write(f"djitellopy=={djitellopy_version}\n")

print("Versions saved to requirements.txt")
