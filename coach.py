import torch
import random
import gym
import time
import numpy as np

class Coach():
    def __init__(self, reward_shaping=None, transition=None):
        super(Coach, self).__init__()
        self.init_frameskip = 1
        self.frameskip = 1
        self.init_action = 0
        self.reward_shaping = reward_shaping
        self.update_interval = 1000
        self.total_step = 0
        self.GAMMA = 0.99
        self.min = 10000
        self.cap = 100000
        self.transition = transition

    def run_episode(self, agent, env, memory, episode, preprocess, epsilon, test, learn):
        done = False
        score = 0.0
        step = 0

        # reset game
        state, info = env.reset()
        state = preprocess(state)
        render = test
        if test:
            epsilon = 0.0

        print("Episode: {} Epsilon:{}".format(episode, epsilon))

        # initial framekskip to get lives
        for i in range(self.init_frameskip):
            _, _, _, _, info = env.step(self.init_action)
            if 'ale.lives' in info:
                lives = info["ale.lives"]

        # episode loop
        while not done:
            y = agent.act(state)

            if torch.rand(1) > epsilon:
                action = int(torch.argmax(y))
            else:
                action = env.action_space.sample()

            reward = 0
            for i in range(self.frameskip):
                next_state, r, done, _, info = env.step(action)
                reward += r
            next_state = preprocess(next_state)

            #print(step, score)
            #reward = max(min(reward, 1.0),-1.0)

            lost_life = False if not done else True

            if 'ale.lives' in info and info["ale.lives"] < lives:
                lives = info["ale.lives"]
                lost_life = True

            score += reward
            self.total_step += 1
            step += 1

            if not self.reward_shaping == None:
                reward = self.reward_shaping(reward, done)

            # prime next state
            if not test:
                memory.append(self.transition(state, action, reward, next_state, not lost_life))
                if len(memory) > self.cap:
                    del memory[0]
                if len(memory) > self.min:
                    learn()

            state = next_state

        print("Episode: {} Score: {}".format(episode, score))
        return memory, score, step
