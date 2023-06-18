from djitellopy import Tello
import cv2
import time
import os

# Define frame size as a tuple
FRAME_SIZE = (320, 240)
IMG_DIR = 'images/'

def process_frame(frame):
    # Rotate the frame and crop around the center
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = frame[frame.shape[0] - FRAME_SIZE[0] : frame.shape[0], 
                  frame.shape[1] - FRAME_SIZE[1] : frame.shape[1]]

    # Display the frame
    cv2.imshow("drone", frame)
    cv2.waitKey(1)
    
    return frame

def main():
    # Initialize the drone and connect
    drone = Tello()
    drone.connect()

    # Set video direction to downward and start streaming
    drone.set_video_direction(1)
    drone.streamon()
    time.sleep(2)

    image_num = 0

    while True:
        # Retrieve and process current frame
        frame_read = drone.get_frame_read()
        processed_frame = process_frame(frame_read.frame)

        key = cv2.waitKey(10)

        # Check for 'esc' key to break the loop
        if key == 27:
            break

        # Check for 's' key to save the image
        elif key == ord('s'):
            # Check if the directory exists, if not create one
            if not os.path.exists(IMG_DIR):
                print("Creating folder")
                os.makedirs(IMG_DIR)

            # Save the image and increment the counter
            cv2.imwrite(os.path.join(IMG_DIR, f'img{image_num}.png'), processed_frame)
            print("Image saved!")
            image_num += 1
            time.sleep(1)

    # Close the video stream and destroy all windows
    drone.streamoff()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
