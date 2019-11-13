#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import csv
import sys

import pandas
import cv2
import numpy as np
from cv_bridge import CvBridge

import rosbag

js_topic = "/pwm_radio_arduino/radio_pwm"
image_topic = "/raspicam_node/image/compressed"


def extract_data(rosbag_path, dir_path):
    bag = rosbag.Bag(rosbag_path, 'r')
    with open(path_steering(dir_path), 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(["timestamp",
                         "msg.steering", "msg.throttle"])
        for topic, msg, t in bag.read_messages(topics=[js_topic]):
            writer.writerow(
                [t.to_nsec(), msg.steering, msg.throttle])

    bridge = CvBridge()
    counter = 0
    with open(path_image_list(dir_path), 'w') as writeFile:
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


def path_image_list(dir_path):
    return os.path.join(dir_path, 'data_image.csv')


def path_steering(dir_path):
    return os.path.join(dir_path, 'data_js.csv')


def match_data(data_dir, dst_csv_path):
    steering_data = pandas.read_csv(path_steering(data_dir))
    image_data = pandas.read_csv(path_image_list(data_dir))

    steering_data = steering_data.sort_values("timestamp")
    image_data = image_data.sort_values("timestamp")

    relative_dir_of_images = os.path.relpath(data_dir, start=os.path.dirname(dst_csv_path))

    with open(dst_csv_path, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow((
                "image_timestamp",
                "steering_timestamp",
                "image",
                "msg.steering",
                "msg.throttle"
            ))
        iter = steering_data.iterrows()
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

            writer.writerow((
                image_row["timestamp"],
                data["timestamp"],
                os.path.join(relative_dir_of_images, image_row["image"]),
                data["msg.steering"],
                data["msg.throttle"]
            ))


def main(in_rosbag, out_csv, out_dir, overwrite):
    """
    :param in_rosbag: location of the input rosbag
    :param out_csv: location of the output csv with relative paths to images and steering
    :param out_dir: directory to write images
    :param overwrite: flag, if the data should be overwritten. The function raises IOError if output exists and
        overwrite is false
    """
    if os.path.exists(out_csv) and not overwrite:
        raise IOError("Error: output csv exists")
    if os.path.exists(out_dir):
        if not overwrite:
            raise IOError("Error: output directory exists")
    else:
        os.makedirs(out_dir)
    extract_data(in_rosbag, out_dir)
    match_data(out_dir, out_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('-rb', type=str, required=True, help='path to the rosbag')
    parser.add_argument('-d', type=str, required=True, help='directory to store the data')
    parser.add_argument('--csv', '-o', type=str, required=False, help='path to the compiled csv')
    parser.add_argument('--overwrite', '-y', action='store_true', help='overwrite if directory exists')
    args = parser.parse_args()
    dst_csv_path = args.csv if args.csv is not None else os.path.join(args.d, 'data.csv')

    try:
        main(args.rb, dst_csv_path, args.d, args.overwrite)
    except IOError as e:
        print("Error: {}".format(e), file=sys.stderr)
        exit(1)
