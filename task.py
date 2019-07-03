#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import argparse
import os
import sys

import keras
import tensorflow as tf
from flask import Flask,Response,stream_with_context

#import keras_retinanet.bin
#__package__ = "keras_retinanet.bin"

flask_app = Flask(__name__)
app_config=[]

# Allow relative imports when being executed as script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
#print("1",sys.path)

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet import models
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.preprocessing.pascal_voc import PascalVocGenerator
from keras_retinanet.utils.eval import evaluate,_get_detections_video
from keras_retinanet.utils.keras_version import check_keras_version
import cv2

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_generator():
    validation_generator = CSVGenerator(
        "./wurenji/helmet/pre/test.csv",
        "./wurenji/helmet/pre/class.csv",
    )
    return validation_generator

def eval(video_path="./wurenji/helmet/video/2.mp4"):
    

    # make sure keras is the minimum required version
    check_keras_version()

    keras.backend.tensorflow_backend.set_session(get_session())
    save_path = "./wurenji/helmet/test_result/"
    max_detections = 100 
    score_threshold = 0.05
    iou_threshold = 0.5
    
    # make save path if it doesn't exist
    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)

    # create the generator
    generator = create_generator()

    b = "resnet101"
    m = "./wurenji/helmet/cvth5/resnet101_csv_20.h5"
    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(m, backbone=b)

    # print model summary
    print(model.summary())

    # start evaluation
    
    for item in (_get_detections_video(generator,video_path, model, score_threshold=0.05, max_detections=100, save_path=save_path)):
        print(video_path)
        #yield item
        img,c_p,c_n = item
        print("postive,negtative!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",c_p,"\\\\\\\\\\\\\\\\\\\"c_n)
        ret, img = cv2.imencode('.jpg', img)
        frame=img.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
   
@flask_app.route('/')
def index():
    url='rtsp://admin:juancheng1@221.1.215.254:554'
    return Response(stream_with_context(eval(url)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    #print("2",sys.path)
    parser=argparse.ArgumentParser()
    parser.add_argument('-p','--port',help='port for web application',default=5005,type=int)
    args=parser.parse_args()
    flask_app.run(debug=True, host='0.0.0.0', port=args.port)
