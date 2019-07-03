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

# Allow relative imports when being executed as script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
print("e1",sys.path)
import keras_retinanet.bin
__package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models
from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..utils.eval import evaluate,_get_detections_video
from ..utils.keras_version import check_keras_version


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_generator():
    validation_generator = CSVGenerator(
        "../../wurenji/helmet/pre/test.csv",
        "../../wurenji/helmet/pre/class.csv",
    )
    return validation_generator

def eval(video_path="../../wurenji/helmet/video/2.mp4"):
    

    # make sure keras is the minimum required version
    check_keras_version()

    keras.backend.tensorflow_backend.set_session(get_session())
    save_path = "../../wurenji/helmet/test_result/"
    max_detections = 100 
    score_threshold = 0.05
    iou_threshold = 0.5
    
    # make save path if it doesn't exist
    #if save_path is not None and not os.path.exists(save_path):
        #os.makedirs(save_path)

    # create the generator
    generator = create_generator()

    b = "resnet101"
    m = "../../wurenji/helmet/cvth5/resnet101_csv_20.h5"
    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(m, backbone=b)

    # print model summary
    print(model.summary())

    # start evaluation
    
    for item in (_get_detections_video(generator,video_path, model, score_threshold=0.05, max_detections=100, save_path=save_path)):
        print(item)
    
eval()
