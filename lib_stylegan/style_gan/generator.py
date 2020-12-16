import tensorflow as tf
import tensorflow.keras as keras

import lib_stylegan


def to_rgb(inp, style, im_size, nchw):
    current_size = inp.shape[2 + int(nchw)]
    x = lib_stylegan.style_gan.conv_mod.Conv2DMod(3, 1, 
                              kernel_initializer = keras.initializers.VarianceScaling(200/current_size), 
                              demod = False,
                              nchw=nchw)([inp, style])
    factor = im_size // current_size
    x = keras.layers.UpSampling2D(size=(factor, factor), interpolation='bilinear', 
                                  data_format="channels_first" if  nchw else "channels_last")(x)
    return x

def g_block(x, input_style, input_noise, nb_filters, im_size, nchw, upsampling = True):
    input_filters = x.shape[1 if nchw else -1] 
    data_format="channels_first" if  nchw else "channels_last"
    if upsampling:
        x = keras.layers.UpSampling2D(interpolation='bilinear',  
                                  data_format=data_format)(x)
    
    style = keras.layers.Dense(input_filters, kernel_initializer = 'he_uniform')(input_style)
    d = keras.layers.Conv2D(nb_filters, 1, kernel_initializer = 'zeros',  
                                  data_format=data_format)(input_noise)
    x = lib_stylegan.style_gan.conv_mod.Conv2DMod(filters=nb_filters, kernel_size = 3, padding = 'same', 
                                                 kernel_initializer = 'he_uniform',
                                                 nchw=nchw)([x, style])
    x = keras.layers.add([x, d])
    x = keras.layers.LeakyReLU(0.2)(x)

    style = keras.layers.Dense(nb_filters, kernel_initializer = 'he_uniform')(input_style)
    d = keras.layers.Conv2D(nb_filters, 1, kernel_initializer = 'zeros',  
                                  data_format=data_format)(input_noise)
    
    x = lib_stylegan.style_gan.conv_mod.Conv2DMod(filters = nb_filters, kernel_size = 3, padding = 'same', 
                                                 kernel_initializer = 'he_uniform',
                                                 nchw=nchw)([x, style])

    x = keras.layers.add([x, d])
    x = keras.layers.LeakyReLU(0.2)(x)

    current_size = x.shape[2]
    rgb_style = keras.layers.Dense(nb_filters, kernel_initializer = keras.initializers.VarianceScaling(200/current_size))(input_style)
    return x, to_rgb(x, rgb_style, im_size, nchw)

def make_style_map(model):
    '''
    The network that will map the noise z to w.
    '''
    S = keras.models.Sequential()
    S.add(keras.layers.Dense(model.latent_size, input_shape=[model.latent_size]))
    for i in range(model.nb_style_mapper_layer):
        S.add(keras.layers.LeakyReLU(0.2))
        S.add(keras.layers.Dense(model.latent_size))
    return S

def make_generator(model):
    '''
    Make stylegan2 generator
    '''
    start_dim = model.im_size // (2**(model.n_layers-1))

    inp_seed = keras.layers.Input([start_dim, start_dim, 4*model.channels])
    inp_style = keras.layers.Input([model.n_layers, model.latent_size])
    inp_noise = []

    outs = []

    if model.nchw:
        x = tf.transpose(inp_seed, (0,3,1,2))
    else:
        x = inp_seed

    for i, channels_mult in enumerate(model.channels_mult_list[:model.n_layers][::-1]):
        noise_size = model.im_size // (2**(model.n_layers-i-1))
        current_noise = keras.layers.Input([noise_size, noise_size, 1])
        x, r = g_block(x, inp_style[:,i], current_noise, channels_mult * model.channels, model.im_size, model.nchw, upsampling = (i!=0)) 
        inp_noise.append(current_noise) 
        outs.append(r)
    x = keras.layers.add(outs)
    x = x/2 + 0.5 #Use values centered around 0, but normalize to [0, 1], providing better initialization
    if model.nchw:
        x = tf.transpose(x, (0,2,3,1))
    return keras.models.Model(inputs = [inp_seed, inp_style] + inp_noise, outputs = x)
