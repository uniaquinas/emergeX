#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Authors:    Dario Cazzani
"""
from utils import get_variables, linear, AdamOptimizer, CMajorScaleDistribution

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from optparse import OptionParser

# Get the MNIST data
mnist = input_data.read_data_sets('../Data', one_hot=True)

# Parameters
input_dim = mnist.train.images.shape[1]
hidden_layer1 = 1000
hidden_layer2 = 1000
z_dim = 160
learning_rate = 1E-4
beta1 = 0.9
batch_size = 128
epochs = 1000
tensorboard_path = 'tensorboard_plots/'
noise_length = int(input_dim / 5.)

# get audio data
audio_data = CMajorScaleDistribution(input_dim, batch_size)

# p(z|X)
def encoder(x):
    e_linear_1 = tf.nn.relu(linear(x, hidden_layer1, 'e_linear_1'))
    e_linear_2 = tf.nn.relu(linear(e_linear_1, hidden_layer2, 'e_linear_2'))
    z_mu = linear(e_linear_2, z_dim, 'z_mu')
    z_logvar = linear(e_linear_2, z_dim, 'z_logvar')
    return z_mu, z_logvar

def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps

# p(X|z)
def decoder(z):
    d_linear_1 = tf.nn.relu(linear(z, hidden_layer2, 'd_linear_1'))
    d_linear_2 = tf.nn.relu(linear(d_linear_1, hidden_layer1, 'd_linear_2'))
    logits = linear(d_linear_2, input_dim, 'logits')
    prob = tf.nn.sigmoid(logits)
    return prob, logits

def train(options):
    with tf.name_scope('Input'):
        X = tf.placeholder(tf.float32, shape=[None, input_dim])
        X_noisy = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='Input_noisy')
        input_images = tf.reshape(X_noisy, [-1, 28, 28, 1])
        if options.TRAIN_AUDIO:
            # Audio inputs normalization
            X_norm = tf.div(tf.add(X, 1.), 2)

    with tf.name_scope('Latent_variable'):
        z = tf.placeholder(tf.float32, shape=[None, z_dim])

    with tf.variable_scope('Encoder'):
        z_mu, z_logvar = encoder(X)

    with tf.variable_scope('Decoder') as scope:
        z_sample = sample_z(z_mu, z_logvar)
        decoder_output, logits = decoder(z_sample)
        if options.TRAIN_AUDIO:
            # audio input "de-normalization"
            decoder_output_denorm = tf.subtract(tf.multiply(decoder_output, 2.), 1.)
        generated_images = tf.reshape(decoder_output, [-1, 28, 28, 1])
        # Unused, unless you want to generate new data from N~(0, 1)
        scope.reuse_variables()
        # Sampling from random z
        X_samples, _ = decoder(z)

    with tf.name_scope('Loss'):
        if options.TRAIN_AUDIO:
            reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X_norm), 1)
        else:
            reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)
        kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
        # VAE loss
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

    # Optimizer
    train_op, grads_and_vars = AdamOptimizer(vae_loss, learning_rate, beta1)

    # Visualization
    tf.summary.scalar(name='Loss', tensor=vae_loss)
    tf.summary.histogram(name='Sampled variable', values=z_sample)

    for grad, var in grads_and_vars:
        tf.summary.histogram('Gradients/' + var.name, grad)
        tf.summary.histogram('Values/' + var.name, var)

    if options.TRAIN_AUDIO:
        tf.summary.audio(name='Input Sounds', tensor=X_noisy, sample_rate = 16000, max_outputs=3)
        tf.summary.audio(name='Generated Sounds', tensor=decoder_output_denorm, sample_rate = 16000, max_outputs=3)
    else:
        tf.summary.image(name='Input Images', tensor=input_images, max_outputs=10)
        tf.summary.image(name='Generated Images', tensor=generated_images, max_outputs=10)

    summary_op = tf.summary.merge_all()

    step = 0
    init = tf.global_variables_initializer()
    n_batches = int(mnist.train.num_examples / batch_size)
    with tf.Session() as sess:
        sess.run(init)
        try:
            if options.TRAIN_AUDIO:
                writer = tf.summary.FileWriter(logdir=tensorboard_path+'/audio/', graph=sess.graph)
            else:
                writer = tf.summary.FileWriter(logdir=tensorboard_path+'/mnist/', graph=sess.graph)
            for epoch in range(epochs):
                for iteration in range(n_batches):
                    if options.TRAIN_AUDIO:
                        batch_x, _ = audio_data.__next__()
                    else:
                        batch_x, _ = mnist.train.next_batch(batch_size)
                    # generate mask for noisy batch
                    mask = np.ones(input_dim)
                    idx = np.random.randint(input_dim - noise_length)
                    mask[idx:idx+noise_length] = 0.
                    mask = np.tile(mask, (batch_size, 1))
                    noisy_batch = batch_x * mask
                    # Train
                    sess.run(train_op, feed_dict={X: batch_x, X_noisy: noisy_batch})

                    if iteration % 10 == 0:
                        summary, batch_loss = sess.run([summary_op, vae_loss], feed_dict={X: batch_x, X_noisy: noisy_batch})
                        writer.add_summary(summary, global_step=step)
                        print("Epoch: {} - Iteration {} - Loss: {:.4f}\n".format(epoch, iteration, batch_loss))

                    step += 1
            print("Model Trained!")

        except KeyboardInterrupt:
            print('Stopping training...')

if __name__ == '__main__':
    parser = OptionParser(usage="usage: %prog [options] filename",
                          version="%prog 0.1")
    parser.add_option("--TRAIN_AUDIO",  default=False,
                          action="store_true",help="Train on audio - otherwise train on MNIST")
    (options, args) = parser.parse_args()
    train(options)
