import json
import os
import tqdm
print(os.environ['TF_CONFIG'])

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import lib_stylegan

def is_chief(task_type, task_id):
    # If `task_type` is None, this may be operating as single worker, which works
    # effectively as chief.
    return task_type is None or task_type == 'chief' or (
        task_type == 'worker' and task_id == 0)


tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])

im_size = 128
latent_size = 256 
channels = 24 # Should be at least 32 for good results
#Chosing the number of layer this way, means we start with 4x4
nb_layer = int(np.log2(im_size) - 1) 

path="/Data/leo/dogs-face-2015/*jpg"

per_worker_batch_size = 16
global_batch_size = per_worker_batch_size * num_workers


strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
task_type, task_id = (strategy.cluster_resolver.task_type,
                      strategy.cluster_resolver.task_id)
global_batch_size = per_worker_batch_size * num_workers

with strategy.scope():
    model = lib_stylegan.style_gan.StyleGan(im_size=im_size, 
                                            latent_size=latent_size, 
                                            channels=channels,
                                            nb_layer=nb_layer,
                                            global_batch_size=global_batch_size)
    model.compile(run_eagerly=True)

dataset = lib_stylegan.dataset.train_dataset(path, 
                                             n_layers=nb_layer, 
                                             im_size=im_size, 
                                             batch_size=per_worker_batch_size,
                                             latent_size=latent_size
                                            )
for args in tqdm.tqdm(dataset.take(100), total=100):
    per_replica_losses = strategy.run(model.train_step, args=(args,))
for args in tqdm.tqdm(dataset.take(1), total=1):
    per_replica_losses = strategy.run(model.train_step, args=(args,))
