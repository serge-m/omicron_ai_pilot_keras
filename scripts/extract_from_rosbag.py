#!/usr/bin/env python
import argparse
import os
import csv

import pandas
import cv2
import numpy as np
from cv_bridge import CvBridge

import rosbag

js_topic = "/pwm_radio_arduino/radio_pwm"
image_topic = "/raspicam_node/image/compressed"


def dir_exists(dir):
    return os.path.exists(dir)


def create_dir(dir):
    os.mkdir(dir)


def extract_data(rosbag_path, dir_path):
    bag = rosbag.Bag(rosbag_path, 'r')
    with open(os.path.join(dir_path, 'data_js.csv'), 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(["timestamp",
                         "msg.steering", "msg.throttle"])
        for topic, msg, t in bag.read_messages(topics=[js_topic]):
            writer.writerow(
                [t.to_nsec(), msg.steering, msg.throttle])

    bridge = CvBridge()
    counter = 0
    with open(os.path.join(dir_path, 'data_image.csv'), 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(["timestamp", "msg.header.seq", "msg.header.stamp.secs", "msg.header.stamp.nsecs", "image"])
        for topic, msg, t in bag.read_messages(topics=[image_topic]):
            img_fname = "frame_{:010d}.jpg".format(counter)
            writer.writerow([t.to_nsec(), msg.header.seq, msg.header.stamp.secs, msg.header.stamp.nsecs,
                             img_fname])
            cv_image = bridge.compressed_imgmsg_to_cv2(msg)
            cv2.imwrite(os.path.join(dir_path, img_fname), cv_image)
            counter += 1

    bag.close()


def match_data(directory):
    print(os.path.join(directory, "data_js.csv"))
    js4_data = pandas.read_csv(os.path.join(directory, "data_js.csv"))
    image_data = pandas.read_csv(os.path.join(directory, "data_image.csv"))

    js4_data.sort_values("timestamp")
    image_data.sort_values("timestamp")

    print(os.path.join(directory, 'data.csv'))
    with open(os.path.join(directory, 'data.csv'), 'w') as writeFile:
        writer = csv.writer(writeFile)
        iter = js4_data.iterrows()
        data = None

        search = True
        for _, image_row in image_data.iterrows():
            while search:
                # in case if we have more images than signals at the end
                try:
                    data = next(iter)[1]
                except:
                    search = False

                if image_row["timestamp"] < data["timestamp"]:
                    break

            writer.writerow([image_row["image"], data["msg.steering"], data["msg.throttle"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('-rb', type=str, required=True, help='path to the rosbag')
    parser.add_argument('-d', type=str, required=True, help='directory to store the data')

    args = parser.parse_args()

    if not dir_exists(args.d):
        create_dir(args.d)

    extract_data(args.rb, args.d)
    match_data(args.d)
