import tensorflow as tf
import numpy as np

# Class that implements the Deep-Q network described in
# "Human-level control through deep reinforcement learning", Mnih et al
# (https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf):
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

        # Policy for x_batch (target network):
        with tf.variable_scope("target_policy") as scope:
            self.t_outputs = self._policy(self.x_batch)

        # Update target policy operation (from the paper every C steps reset Q-hat = Q):
        self.update_target_policy_op = self._update_target_policy()

        # Compute y_j as in the paper:
        self.q_targets = tf.placeholder(tf.float32, shape=[self.batch_size])
        self.action_targets = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size])
        self.done_mask = tf.placeholder(tf.float32, shape=[self.batch_size])
        y_j = self.rewards + self.gamma * tf.multiply(self.done_mask, self.q_targets)

        # Huber loss function:
        action_mask = tf.one_hot(self.action_targets, self.num_actions)
        diff = y_j - tf.reduce_sum(tf.multiply(action_mask, self.q_outputs), axis=1)
        #self.loss = tf.reduce_mean(tf.square(diff)) # squared loss
        self.loss = tf.losses.huber_loss(y_j, tf.reduce_sum(tf.multiply(action_mask, self.q_outputs), axis=1)) # Huber loss with delta=1.0

        # Training step (minimize the loss function):
        self.train_step = optimizer.minimize(self.loss)

    # Define model architecture as described in the paper:
    def _policy(self, x):
        # Values to [-1, 1]:
        x = x / 127.5 - 1

        # Conv + Relu + Conv + Relu + Conv + Relu + FF + FF:
        conv1 = tf.layers.conv2d(x, filters=32, kernel_size=(8, 8), strides=(4, 4), activation=tf.nn.relu, name="conv1")
        conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=(4, 4), strides=(2, 2), activation=tf.nn.relu, name="conv2")
        conv3 = tf.layers.conv2d(conv2, filters=64, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu, name="conv3")
        conv3 = tf.contrib.layers.flatten(conv3)
        ff1 = tf.layers.dense(conv3, units=512, activation=tf.nn.relu, name="ff1")
        ff2 = tf.layers.dense(ff1, units=self.num_actions, name="ff2")

        return ff2

    # Return the tf operation to update the target policy:
    def _update_target_policy(self):
        trainable_vars = tf.trainable_variables()
        q_vars = [v for v in trainable_vars if v.name.startswith('policy/')]
        q_vars.sort(key=lambda x: x.name)
        t_vars = [v for v in trainable_vars if v.name.startswith('target_policy/')]
        t_vars.sort(key=lambda x:x.name)
        return  [v[0].assign(v[1]) for v in zip(t_vars, q_vars)]

    # Execute the operation to update the target policy:
    def update_target_policy(self, sess):
        sess.run(self.update_target_policy_op)

    # Returns Q(x, a; theta) for each action a:
    def predict(self, x, sess):
        return sess.run(self.q_single, feed_dict={self.x_single: x})

    # Returns Q(x_batch, a; theta) for each action a:
    def predict_batch(self, x_batch, sess, use_target_policy=False):
        if use_target_policy:
            return sess.run(self.t_outputs, feed_dict={self.x_batch: x_batch})
        else:
            return sess.run(self.q_outputs, feed_dict={self.x_batch: x_batch})

    # Perform a gradient descent step w.r.t. the network parameters theta (determining the policy):
    def train_batch(self, minibatch_sample, sess):
        (minibatch_state, minibatch_action, minibatch_reward, minibatch_next_state, minibatch_done) = minibatch_sample
        loss, _ = sess.run([self.loss, self.train_step],
                           feed_dict={self.x_batch: minibatch_state,
                                      self.action_targets: minibatch_action,
                                      self.rewards: minibatch_reward,
                                      self.done_mask: np.logical_not(minibatch_done),
                                      self.q_targets: np.amax(self.predict_batch(minibatch_next_state, sess, use_target_policy=True), axis=1)
                                      })
