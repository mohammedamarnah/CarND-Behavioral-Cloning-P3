
import argparse
import base64
import json
import socketio
import eventlet
import eventlet.wsgi
import time

import utils

import numpy as np

from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

MAX_SPEED = 25
MIN_SPEED = 10

speed_limit = MAX_SPEED

def f(speed, steering_angle):
    global speed_limit
    if speed > speed_limit:
        return MIN_SPEED  # slow down
    else:
        return MAX_SPEED
    return 1.0 - steering_angle**2 - (speed/speed_limit)**2

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    
    # The current throttle of the car
    throttle = data["throttle"]
    
    # The current speed of the car
    speed = data["speed"]
    speed = float(speed)

    # The current image from the center camera of the car
    imgString = data["image"]

    ############# Predict the current Steering Angle ####################

    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    image_array = utils.preprocess(image_array)
    image_array = np.array([image_array])

    steering_angle = float(model.predict(image_array, batch_size=1))
    
    ######################################################################

    ############ Calculate the appropriate speed for this frame ##########
    
    pid = utils.PID()
    throttle = pid.calc(speed, f(speed, steering_angle))

    ######################################################################

    print(steering_angle, throttle)

    # Sends the predicted steering angle 
    # and the calculated throttle to the car
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)