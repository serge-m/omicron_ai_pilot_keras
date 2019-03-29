#!/usr/bin/python3

import time

import rospy
from ackermann_msgs.msg import AckermannDriveStamped


class ROSPackage_AI_Driver:
	def __init__(self):
		rospy.init_node('ai_driver')
		
		self.value_to_publish = AckermannDriveStamped()
		self.value_to_publish.drive.speed = 1.0
		self.value_to_publish.drive.steering_angle = 0.0
		self.ai_driver_publisher = rospy.Publisher('ackermann_cmd_mux/input/ai_driver', AckermannDriveStamped, queue_size=10)

	def start(self):
		rate = rospy.Rate(10)
		while not rospy.is_shutdown():
			self.ai_driver_publisher.publish(self.value_to_publish)
			rate.sleep()


if __name__ == '__main__':
	package = ROSPackage_AI_Driver()
	try:
		package.start()
	except rospy.ROSInterruptException:
		pass