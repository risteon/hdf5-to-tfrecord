#!/usr/bin/env python

# Copyright 2017
# ==============================================================================

"""Converts HDF5 data files to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import math

import tensorflow as tf
import numpy as np
import h5py

bar_available = True
try:
    import progressbar
except ImportError:
    bar_available = False

FLAGS = None


def read_hdf5(path):

    sets_to_read = ['point_cloud', 'obj_labels']
    hdf5 = h5py.File(path, "r")
    r = {s: np.array(hdf5[s]) for s in sets_to_read}
    hdf5.close()
    return r


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_array_feature(array):
    return tf.train.Feature(float_list=tf.train.FloatList(value=array))


def _int_array_feature(array):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=array))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_hdf5_files(dataset):

    def get_hdf5_files_in_directory(directory):
        return sorted([os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.hdf5')])

    def get_hdf5_files_from_list(filelist):
        with open(filelist, 'r') as f:
            files = sorted([l for l in f.readlines() if l.endswith('.hdf5')])
        return files

    if os.path.isfile(dataset):
        return get_hdf5_files_from_list(dataset)
    elif os.path.isdir(dataset):
        return get_hdf5_files_in_directory(dataset)
    else:
        raise RuntimeError("Could not retrieve input files from {}".format(dataset))


def convert_to(files, dataset_name, output_dir, samples_per_file=None):

    if not files:
        raise RuntimeError("No files")

    if samples_per_file is None:
        samples_per_file = len(files)

    # count for whole dataset
    unique_values = {}

    # calc number of files
    total = ((len(files) - 1) // samples_per_file) + 1

    # filename formatting
    digits = math.ceil(math.log10(total))
    f_str = '0{}d'.format(digits)
    filename = os.path.join(output_dir, dataset_name + '_{{:{}}}_of_{{:{}}}.tfrecords'.format(f_str, f_str))

    if bar_available:
        bar = progressbar.ProgressBar(redirect_stdout=True, max_value=len(files))

    filelist = []
    processed_counter = 0
    for file_counter in range(total):

        f_name = filename.format(file_counter+1, total)
        filelist.append(f_name)
        print('Writing', f_name)
        with tf.python_io.TFRecordWriter(f_name) as writer:

            for file in files[file_counter * samples_per_file: (file_counter+1)*samples_per_file]:
                try:
                    data = read_hdf5(file)
                except OSError:
                    print("Could not read {}. Skipping.".format(file))
                    continue

                point_cloud = data['point_cloud']
                labels = data['obj_labels']

                # count labels (0, 1, 255)
                u, counts = np.unique(labels, return_counts=True)
                for u, count in zip(u, counts):
                    if u in unique_values:
                        unique_values[u] += count
                    else:
                        unique_values[u] = count

                num_points = point_cloud.shape[0]
                if num_points != labels.shape[0] or num_points != labels.shape[0]:
                    raise RuntimeError("Point cloud size does not match label size in {} ({} vs. {})"
                                       .format(file, point_cloud.shape[0], labels.shape[0]))

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'points': _float_array_feature(point_cloud.flatten()),
                            'label': _int_array_feature(labels),
                        }
                    )
                )
                writer.write(example.SerializeToString())

                if bar_available:
                    bar.update(processed_counter)
                processed_counter += 1

    # write filelist
    f_list_name = os.path.join(output_dir, 'tf_dataset_{}.txt'.format(dataset_name))
    with open(f_list_name, 'w') as file:
        for f in filelist:
            file.write(f + '\n')

    print("List of files written to {}".format(f_list_name))
    print("Unique values in dataset '{}': {}".format(dataset_name, unique_values))


def main(unused_argv):

    if FLAGS.output:
        output_dir = FLAGS.output
    else:
        output_dir = os.getcwd()

    samples_per_file = None
    if FLAGS.filesize:
        samples_per_file = FLAGS.filesize

    for counter, dataset in enumerate(FLAGS.datasets):
        files = get_hdf5_files(dataset)
        if not files:
            print("No data in {}. Skipping.".format(dataset))
            continue
        foldername = '{:02d}'.format(counter)
        folderpath = os.path.join(output_dir, foldername)
        try:
            os.makedirs(folderpath)
        except FileExistsError:
            pass
        convert_to(files, dataset_name=foldername, output_dir=folderpath, samples_per_file=samples_per_file)


if __name__ == '__main__':

    def is_valid_folder(x):
        """
        'Type' for argparse - checks that file exists but does not open.
        """
        x = os.path.expanduser(x)
        if not os.path.isdir(x):
            raise argparse.ArgumentTypeError("{0} is not a directory".format(x))
        return x

    # allow multiple datasets: either path to folders
    parser = argparse.ArgumentParser()
    parser.add_argument('--filesize', metavar='N', type=int)
    parser.add_argument('--output', metavar='directory', type=is_valid_folder)
    parser.add_argument('datasets', nargs='+')

    FLAGS = parser.parse_args()
    tf.app.run(main=main, argv=sys.argv)
