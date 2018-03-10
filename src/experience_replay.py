import numpy as np

# Class for the experience replay used to train the DQN:
class ExperienceReplay:
    # Initialize the experience replay object with a predefined capacity,
    # state_space of the experience of a certain dtype:
    def __init__(self, capacity):
        self.max_capacity = capacity # maximum capacity of the storage
        self.idx = 0                 # next index where newly added object will be
        self.storage = []            # storage containing tuples (s, a, r, s', done)

    # Add an element to the storage:
    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if self.idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.idx] = data
        self.idx = (self.idx + 1) % self.max_capacity

    # Sample batch_size elements from the storage:
    def sample(self, batch_size):
        minibatch_state, minibatch_action, minibatch_reward, minibatch_next_state, minibatch_done = [], [], [], [], []
        idxs = np.random.randint(len(self.storage), size=min(batch_size, len(self.storage)))
        for idx in idxs:
            data = self.storage[idx]
            (state, action, reward, next_state, done) = data
            minibatch_state.append(np.array(state, copy=False))
            minibatch_action.append(np.array(action, copy=False))
            minibatch_reward.append(np.array(reward, copy=False))
            minibatch_next_state.append(np.array(next_state, copy=False))
            minibatch_done.append(np.array(done, copy=False))

        return minibatch_state, minibatch_action, minibatch_reward, minibatch_next_state, minibatch_done
