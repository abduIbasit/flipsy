import argparse
import os
import cv2
import base64
import numpy as np
import requests
import time
import dotenv
import sys
# from flask import Flask, request, jsonify
dotenv.load_dotenv()

# app = Flask(__name__)

start_time = time.time()  # Initialize the start time

ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
ROBOFLOW_MODEL = os.getenv('ROBOFLOW_MODEL')
ROBOFLOW_SIZE = 640

cumulative_sum = 0

def sum_condition():
    global cumulative_sum
    cumulative_sum += 10
    # print(cumulative_sum)  # Print the total so far
    # sys.stdout.flush() # force flush print statement

# Construct the Roboflow Infer URL
# (if running locally replace https://detect.roboflow.com/ with eg http://127.0.0.1:9001/)
upload_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL,
    "?api_key=",
    ROBOFLOW_API_KEY,
    "&format=json",
    "&stroke=5",
    "&confidence=90"
])

# Get webcam interface via opencv-python
# video = None
# for i in range(5):  # Try a few indices
#     video = cv2.VideoCapture(i)
#     if video.isOpened():
#         break

# if video is None or not video.isOpened():
#     raise Exception("Unable to access the camera.")

cap = cv2.VideoCapture(0)

# @app.route('/infer', methods=['POST'])
def infer():
    # video_url = request.form.get('video_url')

    while True:
        # Read a frame from the video stream
        ret, img = cap.read()

        if not ret:
            break  # Break the loop if no more frames

        # resize the frame:
        # frame = cv2.resize(frame, (640, 480))

        # Do your inference or other processing here...
        
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
        }, stream=False)

        result = resp.json()

        if 'predictions' in result:
            # detection_count = len(result['predictions'])
            sum_condition()  # Call the sum_condition function with the number of detections

            for prediction in result['predictions']:
                print(prediction['class'], ":", prediction['confidence'])
                sys.stdout.flush() # force flush 

        # Parse result image
        # resp = resp.raw
        # image = np.asarray(bytearray(resp.read()), dtype="uint8")
        # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # cumulative_sum = f"{cumulative_sum}"
        # print(f"Cumulative Sum: {cumulative_sum}")

        response = {"score":cumulative_sum}
        # response = jsonify(response) 

        return img


# if __name__ == '__main__':
    # app.run(host='0.0.0.0', debug=True, threaded=True, port=8080)
    # Main loop; infers sequentially until you press "q"
while True:
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    
    # Exit the loop after 120 seconds or if "q" key is pressed
    if elapsed_time > 120 or cv2.waitKey(1) == ord('q'):
        break

    # Synchronously get a prediction from the Roboflow Infer API
    image = infer()
    # And display the inference results
    cv2.imshow('image', image)

    # Print frames per second
    # print((1/(time.time()-start)), " fps")
    print(cumulative_sum)
# Release resources when finished
cap.release()
cv2.destroyAllWindows()