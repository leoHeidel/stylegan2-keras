import datetime

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

import style_gan_3d


def gradient_penalty(samples, output, weight):
    gradients = K.gradients(output, samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    return K.mean(gradient_penalty) * weight

def g_block(x, input_style, input_noise, nb_filters, im_size, upsampling = True):
    input_filters = x.shape[-1]
    if upsampling:
        x = keras.layers.UpSampling2D(interpolation='bilinear')(x)
    
    current_size = x.shape[2]
    rgb_style = keras.layers.Dense(nb_filters, kernel_initializer = keras.initializers.VarianceScaling(200/current_size))(input_style)
    style = keras.layers.Dense(input_filters, kernel_initializer = 'he_uniform')(input_style)
    

    noise_cropped = input_noise[:,:current_size, :current_size] 
    d = keras.layers.Dense(nb_filters, kernel_initializer='zeros')(noise_cropped)

    x = style_gan_3d.conv_mod.Conv2DMod(filters=nb_filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')([x, style])
    x = keras.layers.add([x, d])
    x = keras.layers.LeakyReLU(0.2)(x)

    style = keras.layers.Dense(nb_filters, kernel_initializer = 'he_uniform')(input_style)
    d = keras.layers.Dense(nb_filters, kernel_initializer = 'zeros')(noise_cropped)

    x = style_gan_3d.conv_mod.Conv2DMod(filters = nb_filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')([x, style])
    x = keras.layers.add([x, d])
    x = keras.layers.LeakyReLU(0.2)(x)

    return x, to_rgb(x, rgb_style, im_size)

def d_block(inp, fil, p = True):
    res = keras.layers.Conv2D(fil, 1, kernel_initializer = 'he_uniform')(inp)

    out = keras.layers.Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(inp)
    out = keras.layers.LeakyReLU(0.2)(out)
    out = keras.layers.Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(out)
    out = keras.layers.LeakyReLU(0.2)(out)

    out = keras.layers.add([res, out])

    if p:
        out = keras.layers.AveragePooling2D()(out)

    return out

def to_rgb(inp, style, im_size):
    current_size = inp.shape[2]
    x = style_gan_3d.conv_mod.Conv2DMod(3, 1, kernel_initializer = keras.initializers.VarianceScaling(200/current_size), 
                              demod = False)([inp, style])
    factor = im_size // current_size
    x = keras.layers.UpSampling2D(size=(factor, factor), interpolation='bilinear')(x)
    return x

def get_random_noise(batch_size=8):
    random_noise = tf.random.normal(shape=(batch_size, 8))
    return random_noise[:,:3], random_noise[:,3:6], random_noise[:,6], random_noise[:,7]

class StyleGan(keras.Model):
    def __init__(self, steps = 0, lr = 0.0001, im_size=256, latent_size = 512, 
                 channels=32, channels_mult_list=None, seed_type="3d", log_steps=None):
        super(StyleGan, self).__init__()
        
        self.n_layers = int(np.log2(im_size) - 1) -1
        self.im_size = im_size
        self.latent_size = latent_size
        self.channels = channels
        self.channels_mult_list = channels_mult_list or [1,2,4,6,8,16,32,64]
                
        #Models
        self.D = self.make_discriminator()
        self.S = self.make_style_map()
        self.G = self.make_generator()
        if seed_type == '3d':
            self.SN = self.make_seed_network_3d()
        else:
            assert seed_type == "standard"
            self.SN = self.make_seed_network_standard()
        self.S_SN_opt = keras.optimizers.Adam(lr = lr, beta_1 = 0, beta_2 = 0.999)
        self.G_opt = keras.optimizers.Adam(lr = lr, beta_1 = 0, beta_2 = 0.999)
        self.D_opt = keras.optimizers.Adam(lr = lr, beta_1 = 0, beta_2 = 0.999)

        self.steps = steps        
        self.pl_mean = tf.Variable(0, dtype=tf.float32)
        
        logdir = "logs/train_data/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_steps = log_steps
        if log_steps is not None:
            self.file_writer = tf.summary.create_file_writer(logdir)
    
    def make_seed_network_standard(self):
        start_dim = self.im_size // (2**(self.n_layers-1))
        style_input = inp_style = keras.layers.Input([self.n_layers, self.latent_size])
        x = tf.stop_gradient(style_input)[:,0,:1] * 0 + 1
        x = keras.layers.Dense(start_dim*start_dim*4*self.channels, activation = 'relu', 
                             kernel_initializer = 'random_normal')(x)
        x = keras.layers.Reshape([start_dim, start_dim, 4*self.channels])(x)
        return keras.models.Model(inputs = style_input, outputs = x)

    def make_seed_network_3d(self):
        start_dim = self.im_size // (2**(self.n_layers-1))
        style_input = keras.layers.Input([self.n_layers, self.latent_size])
        
        r = get_random_noise(batch_size=tf.shape(style_input)[0])
        random_view = style_gan_3d.lib_3d.layers.CameraStd()(r)
        rays = style_gan_3d.lib_3d.layers.RayTracer()(random_view)

        hiddens = keras.layers.Dense(self.channels*4,activation="relu")(rays)
        hiddens = keras.layers.Dense(self.channels*4,activation="relu")(hiddens)
        hiddens = keras.layers.Dense(self.channels*4,activation="relu")(hiddens)
        
        feature_map = style_gan_3d.lib_3d.math_3d.to_feature_map(hiddens)
        
        return keras.models.Model(inputs = style_input, outputs = feature_map)
    
    def make_discriminator(self):
        d_input = keras.layers.Input(shape = [self.im_size, self.im_size, 3])
        x = d_input
        
        nb_d_layers = int(np.log2(self.im_size))
        channels_mult = 1
        nb_D_layer = int(np.log2(self.im_size)) - 1 
        for channels_mult in self.channels_mult_list[:nb_D_layer-1]:
            x = d_block(x, channels_mult * self.channels)
        x = d_block(x, self.channels_mult_list[nb_D_layer-1] * self.channels, p = False)
        
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(1, kernel_initializer = 'he_uniform')(x)
        return keras.models.Model(inputs = d_input, outputs = x)

    def make_style_map(self):
        S = keras.models.Sequential()
        S.add(keras.layers.Dense(self.latent_size, input_shape = [self.latent_size]))
        S.add(keras.layers.LeakyReLU(0.2))
        S.add(keras.layers.Dense(self.latent_size))
        S.add(keras.layers.LeakyReLU(0.2))
        S.add(keras.layers.Dense(self.latent_size))
        S.add(keras.layers.LeakyReLU(0.2))
        S.add(keras.layers.Dense(self.latent_size))
        S.add(keras.layers.LeakyReLU(0.2))
        return S
    
    def make_generator(self):
        start_dim = self.im_size // (2**(self.n_layers-1))
        
        inp_seed = keras.layers.Input([start_dim, start_dim, 4*self.channels])
        inp_style = keras.layers.Input([self.n_layers, self.latent_size])
        inp_noise = keras.layers.Input([self.im_size, self.im_size, 1])
    
        outs = []
        x = inp_seed
    
        for i, channels_mult in enumerate(self.channels_mult_list[:self.n_layers][::-1]):
            x, r = g_block(x, inp_style[:,i], inp_noise, channels_mult * self.channels, self.im_size, upsampling = (i!=0))  
            outs.append(r)
        x = keras.layers.add(outs)
        x = keras.layers.Lambda(lambda y: y/2 + 0.5)(x) #Use values centered around 0, but normalize to [0, 1], providing better initialization
        return keras.models.Model(inputs = [inp_seed, inp_style, inp_noise], outputs = x)

    @tf.function
    def tf_train_step(self, images, style1, style2, style2_idx, noise, perform_gp=True, perform_pl=False):
        with tf.GradientTape(persistent=True) as grad_tape:
            #Get style information
            w_1 = self.S(style1)
            w_2 = self.S(style2)
            w_space = tf.repeat(tf.stack([w_1,w_2], axis=1),[style2_idx, self.n_layers-style2_idx],axis=1)
            pl_lengths = self.pl_mean

            #Generate images
            seed = self.SN(w_space)
            generated_images = self.G([seed, w_space, noise])

            #Discriminate
            real_output = self.D(images, training=True)
            fake_output = self.D(generated_images, training=True)

            #Hinge loss function
            gen_loss = K.mean(fake_output)
            divergence = K.mean(K.relu(1 + real_output) + K.relu(1 - fake_output))
            disc_loss = divergence

            if perform_gp:
                #R1 gradient penalty
                disc_loss += gradient_penalty(images, real_output, 10)

            if perform_pl:
                #Slightly adjust W space
                w_space_2 = []
                for i in range(self.n_layers):
                    w_slice = w_space[:,i]
                    std = 0.1 / (K.std(w_slice, axis = 0, keepdims = True) + 1e-8)
                    w_space_2.append(w_slice + K.random_normal(tf.shape(w_slice)) / (std + 1e-8))
                w_space_2 = tf.stack(w_space_2, axis=1)
                #Generate from slightly adjusted W space
                pl_images = self.G([seed, w_space_2,noise], training=True)

                #Get distance after adjustment (path length)
                pl_lengths = K.mean(K.square(pl_images - generated_images), axis = [1, 2, 3])
                if self.pl_mean > 0 :
                    gen_loss += K.mean(K.square(pl_lengths - self.pl_mean))

        #Get gradients for respective areas
        gradients_of_generator = grad_tape.gradient(gen_loss, self.G.trainable_variables + self.S.trainable_variables + self.SN.trainable_variables)
        gradients_of_discriminator = grad_tape.gradient(disc_loss, self.D.trainable_variables)
        del grad_tape

        #Apply gradients
        self.G_opt.apply_gradients(zip(gradients_of_generator, self.G.trainable_variables + self.S.trainable_variables + self.SN.trainable_variables))
        self.D_opt.apply_gradients(zip(gradients_of_discriminator, self.D.trainable_variables))

        return disc_loss, gen_loss, divergence, pl_lengths
    
    def train_step(self, args):
        images, style1, style2, style2_idx, noise = args
        self.steps += 1
        
        apply_gradient_penalty = self.steps % 2 == 0 or self.steps < 10000
        apply_path_penalty = self.steps % 16 == 0
        
        disc_loss, gen_loss, divergence, pl_lengths = self.tf_train_step(images, style1, style2, 
                                                                         style2_idx, noise, 
                                                                         apply_gradient_penalty, 
                                                                         apply_path_penalty)
        
        if self.pl_mean == 0:
            self.pl_mean.assign(tf.reduce_mean(pl_lengths))
        self.pl_mean.assign(0.99*self.pl_mean + 0.01*tf.reduce_mean(pl_lengths))
        
        if self.log_steps and not self.steps % self.log_steps:
            with self.file_writer.as_default():
                noise = noise_image(9)
                l_z = latent_z(9)
                l_w = self.S(l_z)
                style = tf.stack([l_w for i in range(n_layers)],axis=1)
                seed = self.SN(style)
                generated = self.G([seed, style, noise])
                img = tf.concat([tf.concat([generated[3*i+k] for k in range(3)], axis=1) for i in range(3)], axis=0)
                tf.summary.image("Training data", [img], step=self.steps)
        
        return {
            "disc_loss":disc_loss,
            "gen_loss":gen_loss,
            "divergence":divergence,
            "pl_lengths":pl_lengths,
        }