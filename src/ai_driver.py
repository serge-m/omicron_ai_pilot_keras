#!/usr/bin/python3

import time

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import CompressedImage

class ROSPackage_AI_Driver:
    def __init__(self):
        rospy.loginfo("AI driver init")
        print("DUPA")
        rospy.init_node('ai_driver')

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

    def receive_compressed_image(self, img):
        rospy.loginfo("img",img)
        self.ai_driver_publisher.publish(self.value_to_publish)

import sys
if __name__ == '__main__':
    package = ROSPackage_AI_Driver()
    try:
        package.start()
    except rospy.ROSInterruptException:
        pass
