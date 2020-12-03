import datetime

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

import style_gan_3d
from . import generator
from . import discriminator
from . import seed

def gradient_penalty(samples, output, weight):
    gradients = K.gradients(output, samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    return K.mean(gradient_penalty) * weight


def apply_EMA(trained_model, ema_model, beta):
    for trained_layer, ema_layer in zip(trained_model.layers, ema_model.layers):
        new_weights = []
        for trained_w, ema_w in zip(trained_layer.get_weights(), ema_layer.get_weights()):
            new_weights.append(beta*ema_w + (1-beta)*trained_w)
        ema_layer.set_weights(new_weights)

class StyleGan(keras.Model):
    def __init__(self, steps = 0, lr = 0.0001, im_size=256, latent_size = 512, 
                 channels=32, channels_mult_list=None, seed_type="3d",
                 nb_style_mapper_layer=5, ema_beta=0.99):
        super(StyleGan, self).__init__()
        
        self.n_layers = int(np.log2(im_size) - 1) -1
        self.im_size = im_size
        self.latent_size = latent_size
        self.channels = channels
        self.channels_mult_list = channels_mult_list or [1,2,4,6,8,16,32,64]
        self.nb_style_mapper_layer = nb_style_mapper_layer
        self.ema_beta = ema_beta
        self.seed_type = seed_type
        #Models
        self.D = discriminator.make_discriminator(self)
        self.M = generator.make_style_map(self)
        self.G = generator.make_generator(self)
        if seed_type == '3d':
            self.S = seed.make_seed_3d(self)
        else:
            assert seed_type == "standard", f"Unrocognized seed_type: {seed_type}"
            self.S = seed.make_seed_standard(self)

        self.S_opt = keras.optimizers.Adam(lr = lr/10, beta_1 = 0, beta_2 = 0.999)
        self.M_opt = keras.optimizers.Adam(lr = lr/10, beta_1 = 0, beta_2 = 0.999)
        self.G_opt = keras.optimizers.Adam(lr = lr, beta_1 = 0, beta_2 = 0.999)
        self.D_opt = keras.optimizers.Adam(lr = lr, beta_1 = 0, beta_2 = 0.999)

        self.steps = steps        
        self.pl_mean = tf.Variable(0, dtype=tf.float32)
        
        logdir = "logs/train_data/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    

    def init_ema(self):
        self.ema_M = generator.make_style_map(self)
        self.ema_G = generator.make_generator(self)
        if self.seed_type == '3d':
            self.ema_S = seed.make_seed_3d(self)
        else:
            assert seed_type == "standard", f"Unrocognized seed_type: {seed_type}"
            self.ema_S = seed.make_seed_standard(self)

        self.ema_S.set_weights(self.S.get_weights())
        self.ema_M.set_weights(self.M.get_weights())
        self.ema_G.set_weights(self.G.get_weights())

    #@tf.function
    def ema_step(self):
        apply_EMA(self.S,self.ema_S, self.ema_beta)
        apply_EMA(self.M,self.ema_M, self.ema_beta)
        apply_EMA(self.G,self.ema_G, self.ema_beta)

    @tf.function
    def tf_train_step(self, images, style1, style2, style2_idx, noise, perform_gp=True, perform_pl=False):
        with tf.GradientTape(persistent=True) as grad_tape:
            #Get style information
            w_1 = self.M(style1)
            w_2 = self.M(style2)
            w_space = tf.repeat(tf.stack([w_1,w_2], axis=1),[style2_idx, self.n_layers-style2_idx],axis=1)
            pl_lengths = self.pl_mean

            #Generate images
            seed = self.S(w_space)
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
        grad_S, grad_M, grad_G = grad_tape.gradient(gen_loss, (self.S.trainable_variables, self.M.trainable_variables, self.G.trainable_variables))
        grad_D = grad_tape.gradient(disc_loss, self.D.trainable_variables)
        del grad_tape

        #Apply gradients
        self.S_opt.apply_gradients(zip(grad_S, self.S.trainable_variables))
        self.M_opt.apply_gradients(zip(grad_M, self.M.trainable_variables))
        self.G_opt.apply_gradients(zip(grad_G, self.G.trainable_variables))
        self.D_opt.apply_gradients(zip(grad_D, self.D.trainable_variables))

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

        return {
            "disc_loss":disc_loss,
            "gen_loss":gen_loss,
            "divergence":divergence,
            "pl_lengths":pl_lengths,
        }