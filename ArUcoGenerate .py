import cv2
import numpy as np

# ArUco dictionary type
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

# Marker ID you want to generate (change this to 20)
marker_id = 20

# Marker size in pixels
marker_size = 200

# Create the marker image
marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
marker_image = cv2.aruco.drawMarker(dictionary, marker_id, marker_size, marker_image)

# Save the marker image to a file
output_filename = f"aruco_marker_{marker_id}.png"
cv2.imwrite(output_filename, marker_image)

print(f"ArUco marker with ID={marker_id} has been generated and saved as {output_filename}.")
