#from styx_msgs.msg import TrafficLight
import datetime
import cv2
import sys
import os

import tensorflow as tf
import numpy as np
from PIL import Image

import yaml

PATH_TF_MODELS_RESEARCH = "/home/student/github/models/research"
PATH_TF_MODELS_SLIM = "/home/student/github/models/research/slim"
PATH_TF_MODELS_OBJECT_DETECTION = "/home/student/github/models/research/object_detection"

sys.path.append(PATH_TF_MODELS_RESEARCH)
sys.path.append(PATH_TF_MODELS_SLIM)
sys.path.append(PATH_TF_MODELS_OBJECT_DETECTION)
from utils import label_map_util
from utils import visualization_utils as vis_util

from utils import dataset_util


FILE_PREFIX_IMG = "IMG_"
DIR_DATA = "DATA/"

PATH_TEST_IMAGE_FILE = "/home/student/CarND-Capstone/ros/src/tl_detector/DATA/IMG_20180701_104354_0.png"

PATH_TRAINED_GRAPH = "/home/student/github/models/research/object_detection/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb"
PATH_LABEL_MAP = "/home/student/github/models/research/object_detection/data/mscoco_label_map.pbtxt"
NUM_CLASSES = 90
#MAX_COUNT_DATA = 100
MAX_COUNT_DATA = 0

PATH_TRAIN_DATA_DIR = ""
#PATH_YAML = "/home/student/Downloads/train.yaml" #TODO change the path
#PATH_YAML = "C:/Work_BigData/Bosch_Small_Traffic_Lights_Dataset/dataset_train_rgb/train.yaml" #TODO change the path
PATH_YAML = "C:/Work_BigData/Bosch_Small_Traffic_Lights_Dataset/dataset_test_rgb/test.yaml" #TODO change the path
DIR_DATA_WITH_YAML = "C:/Work_BigData/Bosch_Small_Traffic_Lights_Dataset/dataset_test_rgb/"

WIDTH_TRAIN_DATA = 1280
HEIGHT_TRAIN_DATA = 720
NUM_CLASSES_TRAIN_DATA = 14
#PATH_TF_RECORD = "train_data.tfrecords"
PATH_TF_RECORD = "test_data.tfrecords"
DICT_LABEL = { "Green" : 1, "Red" : 2, "GreenLeft" : 3, "GreenRight" : 4,
    "RedLeft" : 5, "RedRight" : 6, "Yellow" : 7, "off" : 8,
    "RedStraight" : 9, "GreenStraight" : 10, "GreenStraightLeft" : 11, "GreenStraightRight" : 12,
    "RedStraightLeft" : 13, "RedStraightRight" : 14 }


import contextlib2
from object_detection.dataset_tools import tf_record_creation_util
NUM_SHARDS = 10

class TFRecordCmd(object):
    def __init__(self):
        #TODO load classifier
        #self.append_required_modules()
        #self.load_label_map()
        self.INPUT_TYPES = { "Bosch", "Sloth"}
        self.IDX_BOSCH = 0
        self.IDX_SLOTH = 1
        self.input_type = self.INPUT_TYPES[IDX_SLOTH]
        pass

    def load_label_map(self):
        self.label_map = label_map_util.load_labelmap(PATH_LABEL_MAP)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
                            max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def test(self):
        image = Image.open(PATH_TEST_IMAGE_FILE)
        (width, height) = image.size
        image_array = np.array(image.getdata()).reshape((height, width, 3)).astype(np.uint8)
        image_one_h_w_c = np.expand_dims(image_array, axis=0)
        (boxes, scores, classes, num) = self.run_detection(image_one_h_w_c)
        vis_util.visualize_boxes_and_labels_on_image_array(image_array,
            np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores),
            self.category_index, use_normalized_coordinates=True, line_thickness=8)
        #image_np
        cv2.imwrite("test.png", image_array)
"""
    def write_tf_record(self, path_tf_record, path_yaml, dir_yaml_data):
        writer = tf.python_io.TFRecordWriter(path_tf_record)
        examples = yaml.load(open(path_yaml, 'rb').read())
        count = 0
        for example in examples:
            filename = example['path']
            
            filename = os.path.abspath(os.path.join(os.path.dirname(dir_yaml_data), filename))
            #filename = "C:/Work_BigData/Bosch_Small_Traffic_Lights_Dataset/dataset_train_rgb/" + filename
            #print(filename)
            if (not os.path.exists(filename)):
                print(filename, " does not exist.")
                continue
            count = count + 1
            if count > MAX_COUNT_DATA:
                break
            filename = filename.encode()
            with tf.gfile.GFile(filename, 'rb') as fid:
                encoded_image = fid.read()
            image = Image.open(filename)
            (width, height) = image.size
            image_string = np.array(image).tostring() 
            image_format = 'png'.encode()
            xmins = []
            xmaxs = []
            ymins = []
            ymaxs = []
            classes_text = []
            classes = []
            for box in example['boxes']:
                xmins.append(float(box['x_min']/width))
                xmaxs.append(float(box['x_max']/width))
                ymins.append(float(box['y_min']/height))
                ymaxs.append(float(box['y_max']/height))
                classes_text.append(box['label'].encode('utf-8'))
                print("[", box['label'].encode('utf-8'), "]")
                classes.append(int(DICT_LABEL[box['label']]))

            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height' : dataset_util.int64_feature(height),
                'image/width' : dataset_util.int64_feature(width),
                'image/filename' : dataset_util.bytes_feature(filename),
                'image/source_id' : dataset_util.bytes_feature(filename),
                'image/encoded' : dataset_util.bytes_feature(encoded_image),
                'image/format' : dataset_util.bytes_feature(image_format),
                'image/object/bbox/xmin' : dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax' : dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin' : dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax' : dataset_util.float_list_feature(ymaxs),
                #'image/object/class/text' : dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label' : dataset_util.int64_list_feature(classes),
                #'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
            }))
            writer.write(tf_example.SerializeToString())
        writer.close()
"""

    def create_tf_example_bosch(self, example, filename):
        #print(filename)
        filename = filename.encode()
        with tf.gfile.GFile(filename, 'rb') as fid:
            encoded_image = fid.read()
        image = Image.open(filename)
        (width, height) = image.size
        image_string = np.array(image).tostring() 
        image_format = 'png'.encode()
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []
        for box in example['boxes']:
            xmins.append(float(box['x_min']/width))
            xmaxs.append(float(box['x_max']/width))
            ymins.append(float(box['y_min']/height))
            ymaxs.append(float(box['y_max']/height))
            classes_text.append(box['label'].encode('utf-8'))
            print("[", box['label'].encode('utf-8'), "]")
            classes.append(int(DICT_LABEL[box['label']]))

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height' : dataset_util.int64_feature(height),
            'image/width' : dataset_util.int64_feature(width),
            'image/filename' : dataset_util.bytes_feature(filename),
            'image/source_id' : dataset_util.bytes_feature(filename),
            'image/encoded' : dataset_util.bytes_feature(encoded_image),
            'image/format' : dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin' : dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax' : dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin' : dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax' : dataset_util.float_list_feature(ymaxs),
            #'image/object/class/text' : dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label' : dataset_util.int64_list_feature(classes),
            #'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
        }))
        return tf_example


    def create_tf_example_bosch(self, example, filename):
        #print(filename)
        filename = filename.encode()
        with tf.gfile.GFile(filename, 'rb') as fid:
            encoded_image = fid.read()
        image = Image.open(filename)
        (width, height) = image.size
        image_string = np.array(image).tostring() 
        image_format = 'png'.encode()
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []
        for box in example['boxes']:
            xmins.append(float(box['x_min']/width))
            xmaxs.append(float(box['x_max']/width))
            ymins.append(float(box['y_min']/height))
            ymaxs.append(float(box['y_max']/height))
            classes_text.append(box['label'].encode('utf-8'))
            print("[", box['label'].encode('utf-8'), "]")
            classes.append(int(DICT_LABEL[box['label']]))

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height' : dataset_util.int64_feature(height),
            'image/width' : dataset_util.int64_feature(width),
            'image/filename' : dataset_util.bytes_feature(filename),
            'image/source_id' : dataset_util.bytes_feature(filename),
            'image/encoded' : dataset_util.bytes_feature(encoded_image),
            'image/format' : dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin' : dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax' : dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin' : dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax' : dataset_util.float_list_feature(ymaxs),
            #'image/object/class/text' : dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label' : dataset_util.int64_list_feature(classes),
            #'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
        }))
        return tf_example

    def create_tf_example_sloth(self, example, filename):
        #print(filename)
        filename = filename.encode()
        with tf.gfile.GFile(filename, 'rb') as fid:
            encoded_image = fid.read()
        image = Image.open(filename)
        (width, height) = image.size
        image_string = np.array(image).tostring() 
        image_format = 'png'.encode()
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []
        for box in example['annotations']:
            box_x = box['x']
            box_y = box['y']
            box_width = box['width']
            box_height = box['height']
            xmins.append(float(box_x/width))
            xmaxs.append(float((box_x + box_width)/width))
            ymins.append(float(box_y/height))
            ymaxs.append(float((box_y + box_height)/height))
            classes_text.append(box['label'].encode('utf-8'))
            print("[", box['class'].encode('utf-8'), "]")
            classes.append(int(DICT_LABEL[box['class']]))

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height' : dataset_util.int64_feature(height),
            'image/width' : dataset_util.int64_feature(width),
            'image/filename' : dataset_util.bytes_feature(filename),
            'image/source_id' : dataset_util.bytes_feature(filename),
            'image/encoded' : dataset_util.bytes_feature(encoded_image),
            'image/format' : dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin' : dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax' : dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin' : dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax' : dataset_util.float_list_feature(ymaxs),
            #'image/object/class/text' : dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label' : dataset_util.int64_list_feature(classes),
            #'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
        }))
        return tf_example

    def write_tf_record_shard(self, path_tf_record, path_yaml, dir_yaml_data, num_shards):
        with contextlib2.ExitStack() as tf_record_close_stack:
            output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
                tf_record_close_stack, path_tf_record, num_shards) #output_filebase, num_shards)
            examples = yaml.load(open(path_yaml, 'rb').read())
            count = 0
            #for index, example in examples:
            for example in examples:
                #print("index:", index)
                print("example:", example)
                
                key_file = None
                if (self.input_type == self.INPUT_TYPES[IDX_BOSCH]):
                    key_file = 'path'
                else if (self.input_type == self.INPUT_TYPES[IDX_SLOTH]):
                    key_file = 'filename'
                
                filename = example['path']
                filename = os.path.abspath(os.path.join(os.path.dirname(dir_yaml_data), filename))
                #filename = "C:/Work_BigData/Bosch_Small_Traffic_Lights_Dataset/dataset_train_rgb/" + filename
                #print(filename)
                if (not os.path.exists(filename)):
                    print(filename, " does not exist.")
                    continue
                count = count + 1
                if MAX_COUNT_DATA != 0 and count > MAX_COUNT_DATA:
                    break
                tf_example = None
                if (self.input_type == self.INPUT_TYPES[IDX_BOSCH]):
                    tf_example = self.create_tf_example_bosch(example, filename)
                else if (self.input_type == self.INPUT_TYPES[IDX_SLOTH]):
                    tf_example = self.create_tf_example_sloth(example, filename)
                #output_shard_index = index % num_shards
                output_shard_index = count % num_shards
                output_tfrecords[output_shard_index].write(tf_example.SerializeToString())

    def confirm_tf_record(self, path_tf_record):
        file_name_queue = tf.train.string_input_producer([path_tf_record])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_name_queue)
        features = tf.parse_single_example(serialized_example, features = {
                #"class_count": tf.FixedLenFeature([], tf.int64),
                #"image/object/class/label": tf.FixedLenFeature([], tf.int64),
                "image": tf.FixedLenFeature([], tf.string),
                "image/height": tf.FixedLenFeature([], tf.int64),
                #"image/width": tf.FixedLenFeature([], tf.int64),
                #"depth": tf.FixedLenFeature([], tf.int64),
                })
        for feature in features:
            print("Feature")
        
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                height = tf.cast(features["image/height"], tf.int64).eval()
                #width = tf.cast(features["image/width"], tf.int64).eval()
                #depth = tf.cast(features["depth"], tf.int32).eval()
                #class_count = tf.cast(features["class_count"], tf.int32).eval()
                print("Height:", height)
                #label = tf.cast(features["image/object/class/label"], tf.int64)
                #img = tf.reshape(tf.decode_raw(features["image"], tf.uint8),
                #           tf.stack([height, width, depth]))
            finally:
                coord.request_stop()
            coord.join(threads)

    def train_batch(self):
        img = tf.cast(img, tf.float32) * (1. / 255)
        label = tf.cast(label, dtype=tf.float32)
        batch_size = 100
        # https://www.tensorflow.org/api_docs/python/tf/train/batch
        images, sparse_labels = tf.train.batch([img, label], batch_size=batch_size,
                num_threads=2,
                capacity=1000 + 3 * batch_size)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            images, lobels = sess.run([img, label])# TODO 
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    tf_record_cmd = TFRecordCmd()
    if (True):#not os.path.exists(PATH_TF_RECORD)):
        tf_record_cmd.write_tf_record_shard(PATH_TF_RECORD, PATH_YAML, DIR_DATA_WITH_YAML, NUM_SHARDS)
    else:
        tf_record_cmd.confirm_tf_record(PATH_TF_RECORD) #train(PATH_TF_RECORD, PATH_TRAINED_GRAPH)
