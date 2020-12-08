import os
import tempfile

import numpy as np

import lib_stylegan

test_datset_path = "lib_stylegan/test/test_dataset/*.jpg"

def test_dataset():
    dataset = lib_stylegan.dataset.train_dataset(test_datset_path, im_size=128, batch_size=8)
    for args in dataset.take(1):
        pass
    
def test_style_gan_compiling():
    model = lib_stylegan.style_gan.StyleGan()
    model.compile(run_eagerly=True)


def get_small_params():
    model_param = {
       "im_size" : 64,
       "latent_size" : 64,
       "channels" : 8
       }

    dataset_param = {
        "batch_size" : 3,
        "im_size" : model_param["im_size"],
    }

    return model_param, dataset_param

def test_small_style_gan_fit():
    model_param, dataset_param = get_small_params()

    model = lib_stylegan.style_gan.StyleGan(**model_param)
    model.compile(run_eagerly=True)
    dataset = lib_stylegan.dataset.train_dataset(test_datset_path, **dataset_param)
    model.fit(dataset.take(20))
        
def test_small_style_gan_fit_eager():
    model_param, dataset_param = get_small_params()

    model = lib_stylegan.style_gan.StyleGan(**model_param)
    model.compile(run_eagerly=True)
    dataset = lib_stylegan.dataset.train_dataset(test_datset_path, **dataset_param)
    model.fit(dataset.take(20))
        
def test_3d_seed():
    model_param, dataset_param = get_small_params()
    
    model = lib_stylegan.style_gan.StyleGan(seed_type = "3d", **model_param)
    model.compile()
    dataset = lib_stylegan.dataset.train_dataset(test_datset_path, **dataset_param)
    model.fit(dataset.take(20))
    
def test_save_weights():
    model_param, dataset_param = get_small_params()

    model = lib_stylegan.style_gan.StyleGan(**model_param)
    model.compile()
    dataset = lib_stylegan.dataset.train_dataset(test_datset_path, **dataset_param)
    model.fit(dataset.take(1))

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "tmp_model.tf")
        model.save_weights(model_path)
        model2 = lib_stylegan.style_gan.StyleGan(**model_param)
        model.load_weights(model_path)

def test_ema():
    model_param, dataset_param = get_small_params()

    model = lib_stylegan.style_gan.StyleGan(**model_param)
    model.compile()
    dataset = lib_stylegan.dataset.train_dataset(test_datset_path, **dataset_param)
    model.fit(dataset.take(1))

    model.init_ema()
    model.ema_step()
    model.ema_step()