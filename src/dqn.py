import tensorflow as tf
import numpy as np

# Class that implements the Deep-Q network described in
# "Playing Atari with Deep Reinforcement Learning" by Mnih et al.
# (https://arxiv.org/pdf/1312.5602.pdf):
class DQN:
    def __init__(self,
                 batch_size,
                 state_shape,
                 num_actions,
                 optimizer,
                 gamma=0.99):
        self.batch_size  = batch_size
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma

        # Define computational graph as in the paper:
        # Placeholder for input, q targets and action targets to make the mask later:
        self.x_single = tf.placeholder(tf.float32, shape=[1] + self.state_shape)
        self.x_batch  = tf.placeholder(tf.float32, shape=[self.batch_size] + self.state_shape)

        # Policy for x_single and x_batch:
        with tf.variable_scope("policy") as scope:
            self.q_single  = self._policy(self.x_single)
            scope.reuse_variables()
            self.q_outputs = self._policy(self.x_batch)

        self.q_targets = tf.placeholder(tf.float32, shape=[self.batch_size])
        self.action_targets = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size])
        self.done_mask = tf.placeholder(tf.float32, shape=[self.batch_size])
        y_j = self.rewards + self.gamma * tf.multiply(self.done_mask, self.q_targets)

        # Loss function (see paper eq. 2), L_i(theta_i) = E((y_i - Q(s, a; theta_i))^2):
        action_mask = tf.one_hot(self.action_targets, self.num_actions)
        diff = y_j - tf.reduce_sum(tf.multiply(action_mask, self.q_outputs), axis=1)
        self.loss = tf.reduce_mean(tf.square(diff))

        # Training step (minimize the loss function):
        self.train_step = optimizer.minimize(self.loss)

    def _policy(self, x):
        # Values to [-1, 1]:
        x = x / 127.5 - 1

        # Conv + Relu + Conv + Relu + FF + FF:
        conv1 = tf.layers.conv2d(x, filters=16, kernel_size=(8, 8), strides=(4, 4), activation=tf.nn.relu, name="conv1")
        conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=(4, 4), strides=(2, 2), activation=tf.nn.relu, name="conv2")
        conv2 = tf.contrib.layers.flatten(conv2)
        ff1 = tf.layers.dense(conv2, units=512, activation=tf.nn.relu, name="ff1")
        ff2 = tf.layers.dense(ff1, units=self.num_actions, name="ff2")

        return ff2

    # Returns Q(x, a; theta) for each action a:
    def predict(self, x, sess):
        return sess.run(self.q_single, feed_dict={self.x_single: x})

    # Returns Q(x_batch, a; theta) for each action a:
    def predict_batch(self, x_batch, sess):
        return sess.run(self.q_outputs, feed_dict={self.x_batch: x_batch})

    def train_batch(self, minibatch_sample, sess):
        (minibatch_state, minibatch_action, minibatch_reward, minibatch_next_state, minibatch_done) = minibatch_sample
        loss, _ = sess.run([self.loss, self.train_step],
                           feed_dict={self.x_batch: minibatch_state,
                                      self.action_targets: minibatch_action,
                                      self.rewards: minibatch_reward,
                                      self.done_mask: np.logical_not(minibatch_done),
                                      self.q_targets: np.amax(self.predict_batch(minibatch_next_state, sess), axis=1)
                                      })
