import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import tqdm

import lib_stylegan
lib_stylegan.style_gan.logging.layers.start_logging()

im_size = 128
batch_size = 4
latent_size = 256 #512
channels = 24 # Should be at least 32 for good results
#Chosing the number of layer this way, means we start with 4x4

dataset = lib_stylegan.dataset.train_dataset('/Data/dataset/dogs_full_dataset/*.jpg', 
                                             batch_size=batch_size,
                                            im_size=im_size)


model = lib_stylegan.style_gan.StyleGan(im_size=im_size, 
                                        latent_size=latent_size,
                                        nb_style_mapper_layer=4, #6
                                        channels=channels)

model.compile(run_eagerly=True)
nb_steps = 100
for x in tqdm.tqdm(dataset.take(nb_steps)):
    model.test_step(x)

def to_numpy(d):
    return {k:d[k].numpy() for k in d}

logs = to_numpy(lib_stylegan.style_gan.logging.get_logs())

d_path_mean = []
d_path_std = []
for i in range(model.n_layers-1):
    d_path_mean.append(f"d_block_{i}_0_mean")
    d_path_mean.append(f"d_block_{i}_1_mean")
    d_path_mean.append(f"d_block_{i}_out_mean")
    d_path_std.append(f"d_block_{i}_inp_std")
    d_path_std.append(f"d_block_{i}_inp_conv_std")
    d_path_std.append(f"d_block_{i}_0_std")
    d_path_std.append(f"d_block_{i}_1_std")
    d_path_std.append(f"d_block_{i}_out_std")    
    d_path_std.append(f"d_block_{i}_pooled_std")    

g_path_mean = ["generator_seed_mean"]
g_path_std = ["generator_seed_std"]
rgb_contrib_mean = []
rgb_contrib_std = []
for i in range(model.n_layers):
    g_path_mean.append(f"g_block_{i}_mod_0_mean")
    g_path_mean.append(f"g_block_{i}_noised_0_mean")
    g_path_mean.append(f"g_block_{i}_mod_1_mean")
    g_path_mean.append(f"g_block_{i}_noised_1_mean")
    g_path_std.append(f"g_block_{i}_start_std")
    g_path_std.append(f"g_block_{i}_mod_0_std")
    g_path_std.append(f"g_block_{i}_noised_0_std")
    g_path_std.append(f"g_block_{i}_mid_relu_std")
    g_path_std.append(f"g_block_{i}_mod_1_std")
    g_path_std.append(f"g_block_{i}_noised_1_std")
    g_path_std.append(f"g_block_{i}_finish_std")
    rgb_contrib_mean.append(f"g_block_{i}_rgb_mod_mean")
    rgb_contrib_mean.append(f"g_block_{i}_rgb_mean")
    rgb_contrib_std.append(f"g_block_{i}_rgb_mod_std")
    rgb_contrib_std.append(f"g_block_{i}_rgb_std")
    
logs = pd.Series(logs)
#print("rgb_contrib_mean")
#print("rgb_contrib_std")
#print(logs[rgb_contrib_mean])
#print(logs[rgb_contrib_std])
print("d_path_std")
print(logs[d_path_std])
