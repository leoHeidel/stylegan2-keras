import glob

import numpy as np
import tensorflow as tf

def read_image(im_path, im_size, float16):
    im_file = tf.io.read_file(im_path)
    im = tf.io.decode_jpeg(im_file, channels=3)
    im = tf.image.resize(im, (im_size,im_size))
    im = tf.image.convert_image_dtype(im, tf.float16 if float16 else tf.float32)/255
    im = tf.image.random_flip_left_right(im)
    return im

def train_dataset(path, batch_size=8, im_size=256, float16=False):
    def _read_image(im_path):
        return read_image(im_path, im_size, float16=float16)
    
    nb_train_image = len(glob.glob(path))
    print("Number of train images found:", nb_train_image)

    im_dataset = tf.data.Dataset.list_files(path)
    im_dataset = im_dataset.map(_read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return train_dataset_with_tf_dataset(im_dataset, batch_size)

def train_dataset_with_tf_dataset(im_dataset, batch_size=8):
    im_dataset = im_dataset.batch(batch_size)
    im_dataset = im_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return im_dataset