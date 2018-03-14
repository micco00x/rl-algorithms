import tensorflow as tf
import numpy as np
import gym

import dqn

import atari_wrappers

episodes = 10

# Set up gym environment:
env = gym.make("PongNoFrameskip-v0")
env = atari_wrappers.MaxAndSkipEnv(env)
env = atari_wrappers.wrap_deepmind(env, frame_stack=True)

# Set up dqn (batch_size and optimizer are not used here but they're required by the class):
state_shape = [84, 84, 4]
dqn = dqn.DQN(32, state_shape, env.action_space.n, tf.train.AdamOptimizer(learning_rate=1e-4))

# Checkpoints:
checkpoint = "../tmp/model_1500.ckpt"

#tf.reset_default_graph()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf_saver = tf.train.Saver()
    tf_saver.restore(sess, checkpoint)

    # Test the network for episodes episodes:
    for episode in range(episodes):
        phi_observation = env.reset()
        done = False

        # Performances per episode:
        reward_per_episode = 0

        # Repeat until the end of the episode:
        while not done:
            env.render()

            # Select an action:
            action = np.argmax(dqn.predict(np.array([phi_observation]), sess))

            # Perform the chosen action in the environment:
            phi_observation, reward, done, info = env.step(action)
            reward_per_episode = reward_per_episode + reward

        # Print reward_per_episode on terminal:
        print("Episode=" + str(episode+1) + ", reward=" + str(reward_per_episode))
