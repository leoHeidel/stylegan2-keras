import os
import tempfile

import numpy as np

import style_gan_3d

test_datset_path = "style_gan_3d/test/test_dataset/*.jpg"

def test_dataset():
    dataset = style_gan_3d.dataset.train_dataset(test_datset_path, n_layers=6, im_size=128, batch_size=8)
    for args in dataset.take(1):
        pass
    
def test_style_gan_compiling():
    model = style_gan_3d.style_gan.StyleGan()
    model.compile(run_eagerly=True)
    
def test_small_style_gan_fit():
    im_size = 64
    batch_size = 3
    latent_size = 64 
    
    model = style_gan_3d.style_gan.StyleGan(im_size=im_size, latent_size=latent_size, channels=8)
    model.compile(run_eagerly=True)
    dataset = style_gan_3d.dataset.train_dataset(test_datset_path, n_layers=model.n_layers, 
                                                 im_size=im_size, batch_size=im_size,
                                                 latent_size=latent_size)
    model.fit(dataset.take(20))
    
def test_save_weights():
    im_size = 64
    batch_size = 3
    latent_size = 64 
    channels = 8

    model = style_gan_3d.style_gan.StyleGan(im_size=im_size, latent_size=latent_size, channels=channels)
    model.compile(run_eagerly=True)
    dataset = style_gan_3d.dataset.train_dataset(test_datset_path, n_layers=model.n_layers, 
                                                 im_size=im_size, batch_size=im_size,
                                                 latent_size=latent_size)
    model.fit(dataset.take(1))

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "tmp_model.tf")
        model.save_weights(model_path)
        model2 = style_gan_3d.style_gan.StyleGan(im_size=im_size, latent_size=latent_size, channels=channels)
        model.load_weights(model_path)
