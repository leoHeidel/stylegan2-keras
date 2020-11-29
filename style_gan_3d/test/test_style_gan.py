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
    