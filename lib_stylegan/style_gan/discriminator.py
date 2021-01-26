import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import lib_stylegan
from lib_stylegan.style_gan.logging.layers import SPP_layer

def d_block(inp, fil, name="d_block", p = True):
    res = keras.layers.Conv2D(fil, 1, kernel_initializer = 'he_uniform')(inp)

    out = keras.layers.Conv2D(filters = fil, kernel_size = 3, padding = 'same', 
                              kernel_initializer = 'he_uniform',
                              name=f"{name}_mod_0")(inp)
    out = SPP_layer(name=f"{name}_0")(out)
    out = keras.layers.LeakyReLU(0.2)(out)
    out = keras.layers.Conv2D(filters = fil, kernel_size = 3, padding = 'same', 
                              kernel_initializer = 'he_uniform',
                              name=f"{name}_mod_1")(out)
    out = SPP_layer(name=f"{name}_1")(out)
    out = keras.layers.LeakyReLU(0.2)(out)

    out = keras.layers.add([res, out])
    out = SPP_layer(name=f"{name}_out")(out)

    if p:
        out = keras.layers.AveragePooling2D()(out)

    return out
    
def make_discriminator(model):
    d_input = keras.layers.Input(shape = [model.im_size, model.im_size, 3])
    x = d_input
    
    nb_d_layers = int(np.log2(model.im_size))
    channels_mult = 1
    nb_D_layer = int(np.log2(model.im_size)) - 1 
    for i,channels_mult in enumerate(model.channels_mult_list[:nb_D_layer-1]):
        x = d_block(x, channels_mult * model.channels, name=f"d_block_{i}")
    x = d_block(x, model.channels_mult_list[nb_D_layer-1] * model.channels, p = False)
    
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1, kernel_initializer = 'he_uniform')(x)[:,0]
    return keras.models.Model(inputs = d_input, outputs = x, name="discriminator") 