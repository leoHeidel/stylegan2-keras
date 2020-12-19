import datetime

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

import lib_stylegan
from . import generator
from . import discriminator
from . import seed



def apply_EMA(trained_model, ema_model, beta):
    for trained_layer, ema_layer in zip(trained_model.layers, ema_model.layers):
        new_weights = []
        for trained_w, ema_w in zip(trained_layer.get_weights(), ema_layer.get_weights()):
            new_weights.append(beta*ema_w + (1-beta)*trained_w)
        ema_layer.set_weights(new_weights)


class StyleGan(keras.Model):
    def __init__(self, steps = 0, lr = 0.0001, im_size=256, latent_size = 512, 
                 channels=32, channels_mult_list=None, seed_type="standard",
                 nb_style_mapper_layer=5, ema_beta=0.999, nb_layer=None, 
                 global_batch_size=None, mixed_proba=0.9, random_generator=None,
                 log_dir=None):
        super(StyleGan, self).__init__()
        
        if nb_layer is None:
            if seed_type == "standard":
                #Start with 4*4
                self.n_layers =int(np.log2(im_size) - 1)
            else :
                #Start with 8*8
                self.n_layers =int(np.log2(im_size) - 1) -1
        else :
            self.n_layers = nb_layer 
        self.mixed_proba = mixed_proba
        self.im_size = im_size
        self.latent_size = latent_size
        self.channels = channels
        self.channels_mult_list = channels_mult_list or [1,2,4,6,8,16,32,64]
        self.nb_style_mapper_layer = nb_style_mapper_layer
        self.ema_beta = ema_beta
        self.seed_type = seed_type
        self.global_batch_size = global_batch_size
        self.random_generator = random_generator or tf.random.Generator.from_non_deterministic_state()
        #Models
        self.D = discriminator.make_discriminator(self)
        self.M = generator.make_style_map(self)
        self.G = generator.make_generator(self)
        if seed_type == '3d':
            self.S = seed.make_seed_3d(self)
        else:
            assert seed_type == "standard", f"Unrocognized seed_type: {seed_type}"
            self.S = seed.make_seed_standard(self)

        self.S_opt = keras.optimizers.Adam(lr = lr, beta_1 = 0, beta_2 = 0.999)
        self.M_opt = keras.optimizers.Adam(lr = lr/10, beta_1 = 0, beta_2 = 0.999)
        self.G_opt = keras.optimizers.Adam(lr = lr, beta_1 = 0, beta_2 = 0.999)
        self.D_opt = keras.optimizers.Adam(lr = lr, beta_1 = 0, beta_2 = 0.999)

        self.steps = tf.Variable(steps, dtype=tf.int32)       
        self.pl_mean = tf.Variable(0, dtype=tf.float32)

        self.log_dir = log_dir
        if self.log_dir:
            self.init_tensorboard() 
            

    def init_ema(self):
        self.ema_M = generator.make_style_map(self)
        self.ema_G = generator.make_generator(self)
        if self.seed_type == '3d':
            self.ema_S = seed.make_seed_3d(self)
        else:
            assert self.seed_type == "standard", f"Unrocognized seed_type: {self.seed_type}"
            self.ema_S = seed.make_seed_standard(self)

        self.ema_S.set_weights(self.S.get_weights())
        self.ema_M.set_weights(self.M.get_weights())
        self.ema_G.set_weights(self.G.get_weights())

    def ema_step(self):
        apply_EMA(self.S,self.ema_S, self.ema_beta)
        apply_EMA(self.M,self.ema_M, self.ema_beta)
        apply_EMA(self.G,self.ema_G, self.ema_beta)

    @tf.function
    def tf_train_step(self, images, perform_gp, perform_pl):
        style1, style2, style2_idx, noise = self.get_noise(images)
        with tf.GradientTape(persistent=True) as grad_tape:
            #Get style information
            w_1 = self.M(style1)
            w_2 = self.M(style2)
            stacked = tf.stack([w_1,w_2], axis=1)
            w_space = tf.repeat(stacked,[style2_idx, self.n_layers-style2_idx],axis=1)
            #Generate images
            seed = self.S(w_space)
            generated_images = self.G([seed, w_space, noise])
            
            disc_loss = 0.
            
            #Discriminate, with gradient penalty 
            if perform_gp:
                with tf.GradientTape(watch_accessed_variables=False) as penalty_tape:
                    penalty_tape.watch(images)
                    real_output = self.D(images, training=True)
                gradients = penalty_tape.gradient(real_output, images)
                gradients2 = gradients*gradients
                gradient_penalty = tf.reduce_sum(gradients2, axis=np.arange(1, len(gradients2.shape)))
                disc_loss = disc_loss + 10.*gradient_penalty
                
            else : 
                real_output = self.D(images, training=True)

            fake_output = self.D(generated_images, training=True)
    
            #Hinge loss function
            gen_loss = tf.nn.softplus(-fake_output)
            divergence = tf.nn.softplus(-real_output) + tf.nn.softplus(fake_output)
            disc_loss = disc_loss + divergence

            pl_lengths = self.pl_mean * tf.ones(tf.shape(images)[0])
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
                diff = pl_images - generated_images
                pl_lengths = tf.reduce_mean(diff*diff, axis = [1, 2, 3])
                if self.pl_mean > 0 :
                    diff = pl_lengths - self.pl_mean
                    gen_loss += diff*diff

            gen_loss = tf.nn.compute_average_loss(gen_loss, global_batch_size=self.global_batch_size)
            disc_loss = tf.nn.compute_average_loss(disc_loss, global_batch_size=self.global_batch_size)
            
        #Get gradients for respective areas
        grad_S, grad_M, grad_G = grad_tape.gradient(gen_loss, (self.S.trainable_variables, self.M.trainable_variables, self.G.trainable_variables))
        grad_D = grad_tape.gradient(disc_loss, self.D.trainable_variables)
        del grad_tape

        #Apply gradients
        self.S_opt.apply_gradients(zip(grad_S, self.S.trainable_variables))
        self.M_opt.apply_gradients(zip(grad_M, self.M.trainable_variables))
        self.G_opt.apply_gradients(zip(grad_G, self.G.trainable_variables))
        self.D_opt.apply_gradients(zip(grad_D, self.D.trainable_variables))

        return {
            "disc_loss":disc_loss,
            "gen_loss":gen_loss,
            "divergence":tf.reduce_mean(divergence),
            "pl_lengths":tf.reduce_mean(pl_lengths),
        }
    
    @tf.function 
    def train_step(self, images):
        self.steps.assign(self.steps + 1)
        
        apply_gradient_penalty = ((self.steps % 2) == 0) | (self.steps < 1000)
        apply_path_penalty = self.steps % 16 == 0
        
        losses = self.tf_train_step(images, apply_gradient_penalty, apply_path_penalty)
        pl_lengths = losses["pl_lengths"]
        tf.cond(self.pl_mean == 0, lambda:self.pl_mean.assign(pl_lengths), lambda:pl_lengths)
        self.pl_mean.assign(0.99*self.pl_mean + 0.01*pl_lengths)

        return losses

    def get_noise(self, x):
        '''
        Make a network that will generate the random noised needed for style gan training.
        an input x is needed to give an indication of the batch size.
        generator should be specified if within distributed strategy.
        '''
        batch_size = tf.shape(x)[0]
        noise = [] 
        z_1 = self.random_generator.normal((batch_size, self.latent_size))
        z_2 = self.random_generator.normal((batch_size, self.latent_size))
        only_z2 = tf.cast(self.random_generator.uniform(()) > self.mixed_proba, dtype=tf.int32) # = 0 with proba mixed_prob
        idx = self.random_generator.uniform((), maxval=self.n_layers, dtype=tf.int32) * only_z2

        for i in range(self.n_layers):
            noise_size = self.im_size // (2**(self.n_layers-i-1))
            noise.append(self.random_generator.uniform((batch_size,noise_size,noise_size,1)))

        return z_1, z_2, idx, noise

    def init_tensorboard(self):
        pass

    def tensorboard_step(self, images):
        style1, style2, style2_idx, noise = self.get_noise(images)

        w_1 = self.M(style1)
        w_2 = self.M(style2)
        stacked = tf.stack([w_1,w_2], axis=1)
        w_space = tf.repeat(stacked,[style2_idx, self.n_layers-style2_idx],axis=1)
        #Generate images
        seed = self.S(w_space)
        generated_images = self.G([seed, w_space, noise])
                
        #Gradient Penalty
        with tf.GradientTape() as penalty_tape:
            penalty_tape.watch(images)
            real_output = self.D(images, training=True)
        gradients = penalty_tape.gradient(real_output, images)
        gradients2 = gradients*gradients
        gradient_penalty = tf.reduce_sum(gradients2, axis=np.arange(1, len(gradients2.shape)))
        gradient_penalty = tf.reduce_mean(gradient_penalty)

        fake_output = self.D(generated_images, training=True)

        #Hinge loss function
        gen_loss = fake_output
        divergence = K.relu(1 + real_output) + K.relu(1 - fake_output)
        pl_lengths = self.pl_mean * tf.ones(tf.shape(images)[0])

        ##PL
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
        diff = pl_images - generated_images
        pl_lengths = tf.reduce_mean(diff*diff, axis = [1, 2, 3])
        

        diff = pl_lengths - self.pl_mean
        pl_loss = tf.reduce_mean(diff*diff)
        gen_loss = tf.nn.compute_average_loss(gen_loss, global_batch_size=self.global_batch_size)


        metric = {
            "gen_loss":gen_loss,
            "divergence":tf.reduce_mean(divergence),
            "pl_length":tf.reduce_mean(pl_lengths),
            "pl_loss":pl_loss,
            "pl_length_ema": self.pl_mean,
            "gradient_penalty" : gradient_penalty,
        }