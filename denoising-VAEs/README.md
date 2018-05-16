# Denoising Variational Autoencoders

[![dep2](https://img.shields.io/badge/TensorFlow-1.x%2B-orange.svg)](https://www.tensorflow.org/)
![dep1](https://img.shields.io/badge/Status-Demo-brightgreen.svg)
[![dep3](https://img.shields.io/badge/Cisco-Emerge--AI--Team-yellow.svg)](http://emerge.cisco.com)

![Mnist Reconstruct](https://github.com/ciscoemerge/denoising-VAEs/blob/master/images/denoise-MNIST.png)

### Medium article
Read more on the article [Generating Digits and Sounds with Artificial Neural Nets](https://towardsdatascience.com/generating-digits-and-sounds-with-artificial-neural-nets-ca1270d8445f)

### Requirements
* [x] A decent computer
* [x] TensorFlow 1.x
* [x] Good music

### Train with Digits
```
python train.py
```

### Train with Sounds
```
python train --TRAIN_AUDIO
```

### Visualize on TensorBoard
```
tensorboard --logdir tensorboard_plot
```
