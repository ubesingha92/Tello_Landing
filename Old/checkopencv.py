import cv2

# if hasattr(cv2.aruco, 'estimatePoseSingleMarkers'):
if hasattr(cv2.aruco, 'ARUCO_CCW_CENTER'):
    print("Available")
else:
    print("Not available")
