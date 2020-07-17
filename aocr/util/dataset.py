from __future__ import absolute_import

import logging
import re

import tensorflow as tf

from six import b

import cv2
import os
import random


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def generate(annotations_path, output_path, log_step=5000,
             force_uppercase=True, save_filename=False):

    logging.info('Building a dataset from %s.', annotations_path)
    logging.info('Output file: %s', output_path)

    writer = tf.python_io.TFRecordWriter(output_path)
    longest_label = ''
    idx = 0

    with open(annotations_path, 'r') as annotations:
        for idx, line in enumerate(annotations):
            line = line.rstrip('\n')

            # Split the line on the first whitespace character and allow empty values for the label
            # NOTE: this does not allow whitespace in image paths
            line_match = re.match(r'(\S+)\s(.*)', line)
            if line_match is None:
                logging.error('missing filename or label, ignoring line %i: %s', idx+1, line)
                continue
            (img_path, label) = line_match.groups()

            with open(img_path, 'rb') as img_file:
                img = img_file.read()

            if force_uppercase:
                label = label.upper()

            if len(label) > len(longest_label):
                longest_label = label

            feature = {}
            feature['image'] = _bytes_feature(img)
            feature['label'] = _bytes_feature(b(label))
            if save_filename:
                feature['comment'] = _bytes_feature(b(img_path))

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())

            if idx % log_step == 0:
                logging.info('Processed %s pairs.', idx+1)

    if idx:
        logging.info('Dataset is ready: %i pairs.', idx+1)
        logging.info('Longest label (%i): %s', len(longest_label), longest_label)

    writer.close()

def generate_from_custom(annotations_path, output_path, log_step=5000,
             force_uppercase=True, save_filename=False):

    logging.info('Building a dataset from %s.', annotations_path)
    logging.info('Output dir: %s', output_path)

    train_file = os.path.join(output_path, 'training.tfrecords')
    test_file = os.path.join(output_path, 'testing.tfrecords')

    writerTrain = tf.python_io.TFRecordWriter(train_file)
    writerTest = tf.python_io.TFRecordWriter(test_file)
    longest_label = ''
    idx = 0

    with open(annotations_path, 'r') as annotations:
        for idx, line in enumerate(annotations):
            line = line.rstrip('\n')

            columns = line.split(' ')
            img_path = columns[0]
            img = cv2.imread(img_path)

            # extract every LP
            number_of_lp = int((len(columns)-1 )/ 5)
            for i in range(number_of_lp):
                x = int(columns[i*5+1])
                y = int(columns[i*5+2])
                w = int(columns[i*5+3])
                h = int(columns[i*5+4])
                label = columns[i*5+5]

                if force_uppercase:
                    label = label.upper()
                
                
                lp_img = img[y:y+h, x:x+w, :]
                #lp_img = cv2.cvtColor(lp_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite('tmp.png', lp_img)

                with open('tmp.png', 'rb') as img_file:
                    lp_img = img_file.read()
            
                feature = {}
                feature['image'] = _bytes_feature(lp_img)
                feature['label'] = _bytes_feature(b(label))
                if save_filename:
                    feature['comment'] = _bytes_feature(b(img_path))

                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # send to train or test
                dst = random.choices(['train', 'test'], weights=[0.95, 0.05], k=1)
                if 'train' in dst: writerTrain.write(example.SerializeToString())
                else: writerTest.write(example.SerializeToString())
                

            if idx+number_of_lp % log_step == 0:
                logging.info('Processed %s pairs.', idx+1)

    if idx:
        logging.info('Dataset is ready: %i pairs.', idx+1)

    writerTrain.close()
    writerTest.close()