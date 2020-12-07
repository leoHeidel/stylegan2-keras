import glob

import numpy as np
import tensorflow as tf

def latent_z(batch_size, latent_size):
    return np.random.normal(size=[batch_size, latent_size]).astype('float32')

def noise_image(batch_size, im_size):
    return np.random.uniform(size = [batch_size, im_size, im_size, 1]).astype('float32')


def read_image(im_path, im_size):
    im_file = tf.io.read_file(im_path)
    im = tf.io.decode_jpeg(im_file, channels=3)
    im = tf.image.resize(im, (im_size,im_size))
    im = tf.image.convert_image_dtype(im, tf.float32)/255
    im = tf.image.random_flip_left_right(im)
    return im

def train_dataset(path, n_layers, batch_size=8, im_size=256, latent_size=512, mixed_prob=0.9, random_seed=None):
    def _read_image(im_path):
        return read_image(im_path, im_size)
    
    nb_train_image = len(glob.glob(path))
    print("Number of train images found:", nb_train_image)

    im_dataset = tf.data.Dataset.list_files(path, seed=random_seed)
    im_dataset = im_dataset.map(_read_image)
    return train_dataset_with_tf_dataset(im_dataset, n_layers, batch_size, im_size, latent_size, mixed_prob)

def train_dataset_with_tf_dataset(im_dataset, n_layers, batch_size=8, im_size=256, latent_size=512, mixed_prob=0.9):
    im_dataset = im_dataset.repeat()
    im_dataset = im_dataset.batch(batch_size)
    
    def gen_latent_z():
        while 1:
            yield latent_z(batch_size, latent_size)
 
    def gen_noise():
        while 1:
            yield noise_image(batch_size, im_size)
            
    def gen_mixed_idx():
        while 1:
            if np.random.random() < mixed_prob:
                yield np.random.randint(n_layers)
            else:
                yield n_layers
                       
    latent_z1_dataset = tf.data.Dataset.from_generator(gen_latent_z, tf.float32, output_shapes=(batch_size, latent_size))
    latent_z2_dataset = tf.data.Dataset.from_generator(gen_latent_z, tf.float32, output_shapes=(batch_size, latent_size))
    noise_dataset = tf.data.Dataset.from_generator(gen_noise, (tf.float32))
    mixed_idx_dataset = tf.data.Dataset.from_generator(gen_mixed_idx, (tf.int32))
    
    dataset = tf.data.Dataset.zip((im_dataset, latent_z1_dataset, latent_z2_dataset, 
                                   mixed_idx_dataset, noise_dataset))
    dataset = dataset.prefetch(1)
    return dataset