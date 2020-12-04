import tensorflow as tf
import tensorflow.keras as keras

import lib_stylegan


def get_random_noise(batch_size=8):
    random_noise = tf.random.normal(shape=(batch_size, 8))
    return random_noise[:,:3], random_noise[:,3:6], random_noise[:,6], random_noise[:,7]


def make_seed_standard(model):
        start_dim = model.im_size // (2**(model.n_layers-1))
        style_input = inp_style = keras.layers.Input([model.n_layers, model.latent_size])
        x = tf.stop_gradient(style_input)[:,0,:1] * 0 + 1
        x = keras.layers.Dense(start_dim*start_dim*4*model.channels, activation = 'relu', 
                             kernel_initializer = 'random_normal')(x)
        x = keras.layers.Reshape([start_dim, start_dim, 4*model.channels])(x)
        return keras.models.Model(inputs = style_input, outputs = x)

def make_seed_3d(model):
    start_dim = model.im_size // (2**(model.n_layers-1))
    style_input = keras.layers.Input([model.n_layers, model.latent_size])
    
    r = get_random_noise(batch_size=tf.shape(style_input)[0])
    random_view = lib_stylegan.lib_3d.layers.CameraStd()(r)
    rays = lib_stylegan.lib_3d.layers.RayTracer()(random_view)

    hiddens = keras.layers.Dense(model.channels*4,activation="relu")(rays)
    hiddens = keras.layers.Dense(model.channels*4,activation="relu")(hiddens)
    hiddens = keras.layers.Dense(model.channels*4,activation="relu")(hiddens)
    
    feature_map = lib_stylegan.lib_3d.math_3d.to_feature_map(hiddens)
    
    return keras.models.Model(inputs = style_input, outputs = feature_map)