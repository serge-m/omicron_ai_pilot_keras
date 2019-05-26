import argparse
import os
import csv

import cv2
import numpy as np
from cv_bridge import CvBridge 

import rosbag

js_topic 	= "/low_level/ackermann_cmd_mux/input/teleop"
image_topic = "/raspicam_node/image/compressed"

def dir_exists(dir):
	return os.path.exists(dir)

def create_dir(dir):
	os.mkdir(dir)

def extract_data(rosbag_path, dir_path):
	bag = rosbag.Bag(rosbag_path, 'r')
	with open(os.path.join(dir_path, 'data_js.csv'), 'w') as writeFile:
		writer = csv.writer(writeFile)
		writer.writerow(["timestamp", "msg.header.seq", "msg.header.stamp.secs", "msg.header.stamp.nsecs", "msg.drive.steering_angle", "msg.drive.speed"])
		for topic, msg, t in bag.read_messages(topics=[js_topic]):
			writer.writerow([t.to_nsec(), msg.header.seq, msg.header.stamp.secs, msg.header.stamp.nsecs, msg.drive.steering_angle, msg.drive.speed])

	bridge = CvBridge()
	counter = 0
	with open(os.path.join(dir_path, 'data_image.csv'), 'w') as writeFile:
		writer = csv.writer(writeFile)
		writer.writerow(["timestamp", "msg.header.seq", "msg.header.stamp.secs", "msg.header.stamp.nsecs", "image"])
		for topic, msg, t in bag.read_messages(topics=[image_topic]):
			writer.writerow([t.to_nsec(), msg.header.seq, msg.header.stamp.secs, msg.header.stamp.nsecs, "frame_" + str(counter) + ".png"])
			cv_image = bridge.compressed_imgmsg_to_cv2(msg)
			cv2.imwrite(os.path.join(dir_path, "frame_" + str(counter) +".png"), cv_image)
			counter += 1
	
	bag.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='description')
	parser.add_argument('-rb', type=str, required=True, help='path to the rosbag')
	parser.add_argument('-d', type=str, required=True, help='directory to store the data')

	args = parser.parse_args()

	if(not dir_exists(args.d)):
		create_dir(args.d)

	extract_data(args.rb, args.d)
