import tensorflow as tf
import numpy as np
#import pickle
import gym

import dqn
import experience_replay
import utils

import atari_wrappers

# Extended Data Table 1 | List of hyperparameters and their values:
# (taken from "Human-level control through deep reinforcement learning", Mnih et al.)
episodes = 10000 # note that the paper does not consider episodes but frames
batch_size = 32
replay_memory_size = 100000 # in the paper it's 1M
gamma = 0.99
# using Adam instead of RMSProp as in https://medium.com/mlreview/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55:
#rmsprop_learning_rate = 0.00025
#rmsprop_momentum = 0.95
#rmsprop_epsilon = 0.01
#optimizer = tf.train.RMSPropOptimizer(learning_rate=rmsprop_learning_rate, momentum=rmsprop_momentum, epsilon=rmsprop_epsilon)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
initial_exploration = 1.0
final_exploration = 0.02 # in the paper it's 0.1
final_exploration_frame = 100000 # in the paper it's 1000000
replay_start_size = 10000 # in the paper it's 50K

# Checkpoints:
checkpoint = None
first_episode = 0
first_timestep = 0
checkpoint_folder = "../tmp/"
checkpoint_t = 500
# NOTE: experienceReplay is not going to be saved because Pickle does not
# support big files and because experienceReplay easily reaches high dimensions.
# Possible fix at https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb.
#experience_replay_filename = None

# Performances are saved on a file:
performances = []
performances_file_description = "episode\ttimestep\treward_per_episode"

# Set up gym environment:
env = gym.make("Pong-v0")
env = atari_wrappers.wrap_deepmind(env, frame_stack=True)

# Set up dqn and experienceReplay:
state_shape = [84, 84, 4]
dqn = dqn.DQN(batch_size, state_shape, env.action_space.n, optimizer)
experienceReplay = experience_replay.ExperienceReplay(replay_memory_size)
#if experience_replay_filename:
#    with open(checkpoint_folder + experience_replay_filename, "rb") as experience_replay_file:
#        experienceReplay = pickle.load(experience_replay_file)
#else:
#    experienceReplay = experience_replay.ExperienceReplay(replay_memory_size)

# Set up timestep and epsilon (used in epsilon-greedy strategy):
timestep = first_timestep
epsilon = initial_exploration
eps_greedy_strategy_q = initial_exploration
eps_greedy_strategy_m = (final_exploration - eps_greedy_strategy_q) / final_exploration_frame

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf_saver = tf.train.Saver()

    # Restore TensorFlow checkpoint if specified:
    if checkpoint:
        tf_saver.restore(sess, checkpoint_folder + checkpoint)

    # Train the network for episodes episodes:
    for episode in range(first_episode, episodes):
        phi_observation = env.reset()
        done = False

        # Performances per episode:
        reward_per_episode = 0

        # Repeat until the end of the episode:
        while not done:
            #env.render()

            # Choose an action with epsilon-greedy strategy:
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(dqn.predict(np.array([phi_observation]), sess))

            # Linearly reduce epsilon for the first million timesteps:
            if timestep <= final_exploration_frame:
                epsilon = eps_greedy_strategy_m * timestep + eps_greedy_strategy_q
            timestep = timestep + 1

            # Perform the chosen action in the environment:
            phi_next_observation, reward, done, info = env.step(action)
            reward_per_episode = reward_per_episode + reward

            # Store transition (phi_t, a_t, r_t, phi_t+1, done) in experienceReplay:
            experienceReplay.add(phi_observation, action, reward, phi_next_observation, done)
            phi_observation = phi_next_observation

            # Sample random minibatch of transitions from experienceReplay and
            # perform a gradient descent step on the loss function:
            if timestep >= replay_start_size:
                minibatch_sample = experienceReplay.sample(batch_size)
                loss = dqn.train_batch(minibatch_sample, sess)

        # Print performances per episode:
        print("episode=" + str(episode+1) + ", timestep=" + str(timestep) + ", reward_per_episode=" + str(reward_per_episode))
        performances.append([str(episode+1), str(timestep), str(reward_per_episode)])

        # Save checkpoint and performances:
        if (episode + 1) % checkpoint_t == 0:
            tf_saver.save(sess, checkpoint_folder + "model_" + str(episode+1) + ".ckpt")
            #with open(checkpoint_folder + "experience_replay_" + str(episode+1) + ".pkl", "wb") as experience_replay_file:
            #    pickle.dump(experienceReplay, experience_replay_file, pickle.HIGHEST_PROTOCOL)
            utils.save_performances(checkpoint_folder + "performances_" + str(episode+1) + ".txt", performances, performances_file_description)

# Save performances on a file:
utils.save_performances(checkpoint_folder + "performances.txt", performances, performances_file_description)
