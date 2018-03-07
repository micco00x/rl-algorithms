import tensorflow as tf

# Class that implements the Deep-Q network described in
# "Playing Atari with Deep Reinforcement Learning" by Mnih et al.
# (https://arxiv.org/pdf/1312.5602.pdf):
class DQN:
    def __init__(self,
                 batch_size,
                 state_shape,
                 num_actions,
                 optimizer=tf.train.RMSPropOptimizer(learning_rate=0.00025, momentum=0.95, epsilon=0.01)):
        self.batch_size  = batch_size
        self.state_shape = state_shape
        self.num_actions = num_actions

        # Define computational graph as in the paper:
        # Placeholder for input, q targets and action targets to make the mask later:
        self.x_single  = tf.placeholder(tf.float32, shape=[1] + self.state_shape)
        self.x_batch   = tf.placeholder(tf.float32, shape=[self.batch_size] + self.state_shape)
        self.q_targets = tf.placeholder(tf.float32, shape=[self.batch_size])
        self.action_targets = tf.placeholder(tf.int32, shape=[self.batch_size])

        # Policy for x_single and x_batch:
        self.q_single  = self._policy(self.x_single)
        self.q_outputs = self._policy(self.x_batch)

        # Loss function (see paper eq. 2), L_i(theta_i) = E((y_i - Q(s, a; theta_i))^2):
        action_mask = tf.one_hot(self.action_targets, self.num_actions)
        diff = self.q_targets - tf.reduce_sum(self.q_outputs * action_mask, axis=1)
        self.loss = tf.reduce_mean(tf.square(diff))

        # Training step (minimize the loss function):
        self.train_step = optimizer.minimize(self.loss)

    def _policy(self, x):
        # Values to [-1, 1]:
        x = x / 127.5 - 1

        # Conv + Relu + Conv + Relu + FF + FF:
        conv1 = tf.layers.conv2d(x, filters=16, kernel_size=(8, 8), strides=(4, 4), activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=(4, 4), strides=(2, 2), activation=tf.nn.relu)
        conv2 = tf.contrib.layers.flatten(conv2)
        ff1 = tf.layers.dense(conv2, units=256, activation=tf.nn.relu)
        ff2 = tf.layers.dense(ff1, units=self.num_actions)
        return ff2

    # Returns Q(x, a; theta) for each action a:
    def predict(self, x, sess):
        return sess.run(self.q_single, feed_dict={self.x_single: x})

    # Returns Q(x_batch, a; theta) for each action a:
    def predict_batch(self, x_batch, sess):
        return sess.run(self.q_outputs, feed_dict={self.x_batch: x_batch})

    # Train the network on a single batch and return the loss:
    def train_batch(self, x_batch, q_targets, action_targets, sess):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict={self.x_batch: x_batch, self.q_targets: q_targets, self.action_targets: action_targets})
        return loss
