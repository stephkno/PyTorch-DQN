from collections import deque
import random

class Run:
    def __init__(self, params):
        self.memory = deque(maxlen=params["MEMORY_CAP"])
        self.episodes = []
        self.start = 0
        self.episode = 1
        self.highest_score = -9999
        self.update_steps = 0
        self.max_steps = 0
        self.total_steps = 0

    def AddTransition(self, t):
        self.memory.append(t)

    def Sample(self, n, batch_size=128):
        return random.sample(self.memory, batch_size)
