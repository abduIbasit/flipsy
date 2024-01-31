# import os
# import cv2
# import base64
# import requests
# import time
# import dotenv
# import numpy as np
# import sys

# dotenv.load_dotenv()

# start_time = time.time()  # Initialize the start time

# ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
# ROBOFLOW_MODEL = os.getenv('ROBOFLOW_MODEL')
# ROBOFLOW_SIZE = 640

# cumulative_sum = 0

# def sum_condition():
#     global cumulative_sum
#     cumulative_sum += 10

# upload_url = "".join([
#     "https://detect.roboflow.com/",
#     ROBOFLOW_MODEL,
#     "?api_key=",
#     ROBOFLOW_API_KEY,
#     "&format=json",
#     "&stroke=5",
#     "&confidence=90"
# ])

# cap = cv2.VideoCapture(0)

# def infer():
#     ret, img = cap.read()

#     if not ret:
#         raise ValueError("Failed to capture frame from the camera")

#     height, width, channels = img.shape

#     # Check if the image dimensions are valid
#     if width <= 0 or height <= 0:
#         raise ValueError("Invalid image dimensions")

#     scale = ROBOFLOW_SIZE / max(height, width)
#     img = cv2.resize(img, (round(scale * width), round(scale * height)))

#     retval, buffer = cv2.imencode('.jpg', img)
#     img_str = base64.b64encode(buffer)

#     resp = requests.post(upload_url, data=img_str, headers={
#         "Content-Type": "application/x-www-form-urlencoded"
#     }, stream=True).raw

#     image = np.asarray(bytearray(resp.read()), dtype="uint8")
#     image = cv2.imdecode(image, cv2.IMREAD_COLOR)

#     return image

# if __name__ == '__main__':
#     while 1:
#         try:
#             # On "q" keypress, exit
#             if(cv2.waitKey(1) == ord('q')):
#                 break

#             start = time.time()

#             # Synchronously get a prediction from the Roboflow Infer API
#             image = infer()

#             # And display the inference results
#             cv2.imshow('image', image)

#             # Print frames per second
#             print((1 / (time.time() - start)), " fps")

#             sum_condition()

#         except Exception as e:
#             print(f"An error occurred: {e}")
#             break

#     # Release resources when finished
#     cap.release()
#     cv2.destroyAllWindows()

import os
import cv2
import base64
import numpy as np
import requests
import time

ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
ROBOFLOW_MODEL = os.getenv('ROBOFLOW_MODEL')
ROBOFLOW_SIZE = 640

# Construct the Roboflow Infer URL
# (if running locally replace https://detect.roboflow.com/ with eg http://127.0.0.1:9001/)
upload_url = "".join([
    "https://detect.roboflow.com/",
    "bottle-flip/1",
    "?api_key=",
    "9cXZ7rS4stJYyS8g4JCS",
    "&format=image",
    "&stroke=5"
])

# Get webcam interface via opencv-python
video = cv2.VideoCapture(0)

# Infer via the Roboflow Infer API and return the result
def infer():
    # Get the current image from the webcam
    ret, img = video.read()

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    # Get prediction from Roboflow Infer API
    resp = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True).raw

    # Parse result image
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

# Main loop; infers sequentially until you press "q"
while 1:
    # On "q" keypress, exit
    if(cv2.waitKey(1) == ord('q')):
        break

    # Capture start time to calculate fps
    start = time.time()

    # Synchronously get a prediction from the Roboflow Infer API
    image = infer()
    # And display the inference results
    cv2.imshow('image', image)

    # Print frames per second
    print((1/(time.time()-start)), " fps")

# Release resources when finished
video.release()
cv2.destroyAllWindows()