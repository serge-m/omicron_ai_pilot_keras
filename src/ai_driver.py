#!/usr/bin/python3

import time

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import CompressedImage

from keras.models import load_model
from keras_preprocessing.image.utils import load_img, img_to_array

from PIL import Image as pil_image
from io import BytesIO
import numpy as np

class ROSPackage_AI_Driver:
    def __init__(self):
        rospy.init_node('ai_driver')
        rospy.loginfo("AI driver init. Loading neural network...")
        self.model = load_model('/tmp/best_model.h5')
        rospy.loginfo("Neural network loaded, starting service.")

        self.value_to_publish = AckermannDriveStamped()
        self.value_to_publish.drive.speed = 1.0
        self.value_to_publish.drive.steering_angle = 0.0
        self.ai_driver_publisher = rospy.Publisher(
            'ackermann_cmd_mux/input/ai_driver', AckermannDriveStamped, queue_size=10)
        rospy.Subscriber('raspicam_node/image/compressed', CompressedImage, self.receive_compressed_image)

    def start(self):
        rospy.loginfo("AI driver start")
        rospy.spin()
        #rate = rospy.Rate(10)
        #while not rospy.is_shutdown():
            #self.ai_driver_publisher.publish(self.value_to_publish)
            #rate.sleep()

    def image_to_array(self, img):
        return img_to_array(load_img(BytesIO(img), target_size=(32,32)))

    def receive_compressed_image(self, img):
        rospy.loginfo("img" + type(img).__name__)
        rospy.loginfo("img" + str(len(img.data)))
        image = self.image_to_array(img.data)
        image=np.expand_dims(image,0)
        prediction = self.model.predict(image)
        rospy.loginfo(prediction[0])
        value = (np.argmax(prediction[0]) - 2) / 2
        rospy.loginfo(str(value))
	self.value_to_publish.drive.steering_angle = value
        self.ai_driver_publisher.publish(self.value_to_publish)

import sys
if __name__ == '__main__':
    package = ROSPackage_AI_Driver()
    try:
        package.start()
    except rospy.ROSInterruptException:
        pass
