import tensorflow as tf
import numpy as np
#import pickle
import gym

import dqn
import experience_replay
import utils

# Extended Data Table 1 | List of hyperparameters and their values:
# (taken from "Human-level control through deep reinforcement learning", Mnih et al.)
episodes = 10000 # note that the paper does not consider episodes but frames
batch_size = 32
replay_memory_size = 100000 # in the paper it's 1M
history_length = 4
gamma = 0.99
rmsprop_learning_rate = 0.00025
rmsprop_momentum = 0.95
rmsprop_epsilon = 0.01
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

# Set up dqn and experienceReplay:
state_shape = list(utils.pong_state_shape())
state_shape[2] = history_length
dqn = dqn.DQN(batch_size, state_shape, env.action_space.n, tf.train.RMSPropOptimizer(learning_rate=rmsprop_learning_rate, momentum=rmsprop_momentum, epsilon=rmsprop_epsilon))
experienceReplay = experience_replay.ExperienceReplay(replay_memory_size)
#if experience_replay_filename:
#    with open(checkpoint_folder + experience_replay_filename, "rb") as experience_replay_file:
#        experienceReplay = pickle.load(experience_replay_file)
#else:
#    experienceReplay = experience_replay.ExperienceReplay(replay_memory_size)

# Set up timestep and epsilon (used in epsilon-greedy strategy):
timestep = first_timestep
epsilon = 1.0

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf_saver = tf.train.Saver()

    # Restore TensorFlow checkpoint if specified:
    if checkpoint:
        tf_saver.restore(sess, checkpoint_folder + checkpoint)

    # Train the network for episodes episodes:
    for episode in range(first_episode, episodes):
        observation = env.reset()
        phi_observation = np.concatenate([np.zeros([state_shape[0], state_shape[1], state_shape[2]-1]), utils.pong_preprocess(observation)], axis=2)
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
            if timestep <= 1e6:
                timestep = timestep + 1
                epsilon = -0.9 * 1e-6 * timestep + 1

            # Perform the chosen action in the environment:
            next_observation, reward, done, info = env.step(action)
            reward_per_episode = reward_per_episode + reward

            # Store transition (phi_t, a_t, r_t, phi_t+1, done) in experienceReplay:
            phi_next_observation = np.concatenate([phi_observation[:,:,1:], utils.pong_preprocess(next_observation)], axis=2)
            experienceReplay.add(phi_observation, action, reward, phi_next_observation, done)
            phi_observation = phi_next_observation

            # Sample random minibatch of transitions from experienceReplay and
            # perform a gradient descent step on the loss function:
            if timestep >= replay_start_size:
                minibatch_sample = experienceReplay.sample(batch_size)
                (minibatch_state, minibatch_action, minibatch_reward, minibatch_next_state, minibatch_done) = minibatch_sample
                q_targets = minibatch_reward + gamma * np.multiply([0 if d == True else 1 for d in minibatch_done], np.amax(dqn.predict_batch(minibatch_next_state, sess), axis=1))
                action_targets = minibatch_action
                loss = dqn.train_batch(minibatch_state, q_targets, action_targets, sess)

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
