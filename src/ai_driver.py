#!/usr/bin/env python3
import threading

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import CompressedImage

import tensorflow as tf
from tensorflow.python.keras.models import load_model
from keras_preprocessing.image.utils import load_img, img_to_array

from io import BytesIO
import numpy as np
import sys
import os

import tensorflow.keras as K


class TFModel:
    def __init__(self, path):
        self.path = path
        self.model = load_model(self.path)
        image = np.zeros([1, 120, 160, 3])
        # warm-up run to prepare for multi-threading
        self.model.predict(image)
        self.session = K.backend.get_session()
        self.graph = tf.get_default_graph()
        self.graph.finalize()  # finalize

    def predict(self, image):
        with self.session.as_default():
            with self.graph.as_default():
                prediction = self.model.predict(image)
        return prediction


def create_message(speed, angle):
    message = AckermannDriveStamped()
    message.drive.speed = speed
    message.drive.steering_angle = angle
    return message


class FixedAIDriver:
    def __init__(self, path):
        rospy.init_node('ai_driver')
        self.model = TFModel(path)
        rospy.Subscriber('raspicam_node/image/compressed', CompressedImage, self.receive_compressed_image)
        self.ai_driver_publisher = rospy.Publisher('ackermann_cmd_mux/input/ai_driver', AckermannDriveStamped,
                                                   queue_size=10)

    def start(self):
        rospy.loginfo("AI driver start")
        while not rospy.is_shutdown():
            rospy.spin()
        rospy.loginfo("AI driver finished")

    def image_to_array(self, img):
        return img_to_array(load_img(BytesIO(img), target_size=(120, 160)))

    def receive_compressed_image(self, img):
        rospy.loginfo("PID %d, thread %d ", os.getpid(), threading.get_ident())
        rospy.loginfo("img %s size %d", type(img).__name__, len(img.data))
        image = self.image_to_array(img.data)[np.newaxis]
        prediction = self.model.predict(image)

        rospy.loginfo(prediction[0])
        self.ai_driver_publisher.publish(create_message(speed=0.5 + img.data[12120] * 0.001, angle=prediction[0]))


if __name__ == '__main__':
    try:
        path_model = sys.argv[1]
    except IndexError:
        path_model = '/tmp/model'

    package = FixedAIDriver(path_model)
    package.start()
