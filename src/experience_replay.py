import numpy as np
import random

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
        return random.sample(self.storage, min(batch_size, len(self.storage)))
