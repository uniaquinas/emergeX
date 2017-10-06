import tensorflow as tf
import numpy as np
import sys
import random

def get_variables(shape, scope):
	xavier = tf.contrib.layers.xavier_initializer()
	const = tf.constant_initializer(0.1)
	W = tf.get_variable('weight_{}'.format(scope), shape, initializer=xavier)
	b = tf.get_variable('bias_{}'.format(scope), shape[-1], initializer=const)
	return W, b

def linear(_input, output_dim, scope=None, reuse=None):
	with tf.variable_scope(scope, reuse=reuse):
		shape = [int(_input.get_shape()[1]), output_dim]
		W, b = get_variables(shape, scope)
		return tf.matmul(_input, W) + b

def AdamOptimizer(loss, lr, beta1, var_list=None, clip_grad=False):
	optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1)
	if not var_list:
		grads_and_vars = optimizer.compute_gradients(loss)
	else:
		grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
	if clip_grad:
		grads_and_vars = [(tf.clip_by_norm(grad, 1), var) for grad, var in grads_and_vars]
	train_op = optimizer.apply_gradients(grads_and_vars)
	return train_op, grads_and_vars

def CMajorScaleDistribution(num_samples, batch_size):
	sample_rate = 16000
	seconds = 2
	t = np.linspace(0, seconds, sample_rate*seconds + 1)
	C_major_scale = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]
	while True:
		try:
			batch_x = []
			batch_y = []
			for i in range(batch_size):
				# select random note
				note = C_major_scale[np.random.randint(len(C_major_scale))]
				sound = np.sin(2*np.pi*t*note)
				noise = [random.gauss(0.0, 1.0) for i in range(sample_rate*seconds + 1)]
				noisy_sound = sound + 0.08 * np.asarray(noise)
				start = np.random.randint(0, len(noisy_sound)-num_samples)
				end = start + num_samples
				batch_x.append(noisy_sound[start:end])
				batch_y.append(sound)

			yield np.asarray(batch_x), np.asarray(batch_y)

		except Exception as e:
			print('Could not produce batch of sinusoids because: {}'.format(e))
			sys.exit(1)
