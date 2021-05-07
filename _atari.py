#!/usr/bin/python3
import gym
import torch
import sys
from collections import namedtuple
import random
import time
import copy

env = gym.make("Breakout-ram-v0")

if len(sys.argv) > 1 and sys.argv[1].lower() == "test":
    test = True
else:
    test = False

#### Policy Gradients
# manual seed for random initial weight generation
torch.manual_seed(0)

#hyperparameters
epoch = 0
epochs = 1000
episode = 1
init_action = 1
GAMMA = 0.999
rate = 0.0001
TARGET_INTERVAL = 100
UPDATE_INTERVAL = 100
batch_size = 512

#define neural network model
class Agent(torch.nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 256),
            torch.nn.Tanh(),
        )
        self.head = torch.nn.Linear(256, env.action_space.n)
        self.head.bias.data.fill_(0.0)


    def forward(self, x):
        b = x.shape[0]
        y = self.model(x.float()/255.0)
        return self.head(y.view(b,-1))

def weights_init(m):
    for l in m.modules():
        if type(l) == torch.nn.Linear:
            torch.nn.init.xavier_normal_(l.weight.data,1.0)
            l.bias.data.fill_(1.0)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'value','log_prob'))
#replay memory for episodes
class Memory():
    def __init__(self, batch_size):
        super(Memory, self).__init__()
        self.buffer = []
        self.episode = []
        self.rewards = []
        self.reset()
        self.cap = 10000
        self.n = 1
        self.batch_size = batch_size

    def push(self, *args):
        self.episode.append(Transition(*args))

    def finish_episode(self):
        values = []
        rewards = []
        R = 0.0

        for i, step in enumerate(self.episode):
            reward = step.reward
            R = reward + (GAMMA * R)
            values.append(R)
            rewards.append(reward)

        for i,step in enumerate(self.episode):
            state, action, reward, _, log_prob = step
            self.buffer.append(Transition(state, action, reward, values[i], log_prob))

        while len(self.buffer) > self.cap:
            del self.buffer[0]

        self.episode = []

    def sample(self):
        return random.sample(self.buffer, self.batch_size)

    def get_samples(self):
        return self.buffer

    def reset(self):
        self.buffer = []
        self.length = 0
        self.total_loss = 0

    def learn(self):
        #only update if there are enough samples
        if len(memory.buffer) > batch_size:
            print("\nUpdate Parameters")
            total_loss = 0.0

            for _ in range(self.n):
                #random sample experiences
                #state, actions, reward, values, _ = zip(*memory.sample())
                state, actions, reward, values, _ = zip(*memory.get_samples())
                memory.reset()

                #values = torch.tensor(values)
                #values = values / values.mean()
                #values = values - values.std() + 0.0001
                log_probs =  target_agent.forward(torch.stack(state))
                log_probs = torch.nn.Softmax(dim=1)(log_probs)
                log_probs = torch.gather(log_probs, dim=1, index=torch.tensor(actions).long().unsqueeze(1))

                optimizer.zero_grad()

                for i,log_prob in enumerate(log_probs):
                    loss = -log_prob.log() * values[i]
                    loss.backward(retain_graph=True)
                    total_loss += loss.item()

                print("Loss: {}".format(total_loss))
                optimizer.step()

def preprocess(state):
    return state

def load_agent():
    print("Loading agent")
    os.system("cp /run/user/1000/gvfs/sftp:host=6502.local/Users/stephen/Documents/code/pytorch/reinforement_learning/policy_gradients/checkpoint.pth .")
    agent.load_state_dict(torch.load("./checkpoint.pth"))
    display_params(agent)

def display_params(agent):
    for p in agent.parameters():
        print(p)

memory = Memory(batch_size)
agent = Agent()
weights_init(agent.model)

target_agent = copy.deepcopy(agent)
import os
p_state = 0


#define optimizer with learning rate (gradient step size)
optimizer = torch.optim.Adam(params=agent.parameters(), lr=rate)

def save_model(model):
    print("Saving model.")
    torch.save(model.state_dict(), './checkpoint.pth')

step = 0
highest = 0
print("Initializing replay buffer...")

load_agent()

#run forever
while True:
    done = False

    score = 0.0
    total_score = 0.0
    confidence = 1.0

    #reset game
    state = env.reset()
    state = preprocess(state)

    env._max_episode_steps = 1000
    render = test

    #initial step to get lives info
    _, _, _, info = env.step(init_action)
    if 'ale.lives' in info:
        lives = info["ale.lives"]
    print("Episode {}".format(episode))

    #run episode
    while not done:

        #get Q values
        if test:
            agent.eval()

        dist = torch.nn.Softmax(dim=0)(agent.forward(torch.tensor(state).unsqueeze(0)).view(-1))
        dist = torch.distributions.Categorical(dist)

        action = dist.sample()

        #action = torch.distributions.Categorical(torch.nn.Softmax(dim=-1)(q_values)).sample()
        #take step in environment
        state, reward, done, info = env.step(action)

        state = torch.tensor(state).float()
        state = preprocess(state)


        #check if lost life (only for atari games)
        if 'ale.lives' in info and info["ale.lives"] < lives:
            lives = info["ale.lives"]
            reward = -1.0
            done=True
        else:
            score += reward
            total_score += reward

        if not reward == 0.0 or done:
            memory.finish_episode()

        #render env for observing agent
        if render:
            env.render()

        if step % UPDATE_INTERVAL == 0 and not test:
            memory.learn()
            save_model(agent)
        if step % TARGET_INTERVAL == 0 and not test:
            agent.load_state_dict(target_agent.state_dict())

        #push this step experience to replay memory
        if not test:
            memory.push(state, action, reward, 0.0, dist.log_prob(action))
            step += 1

        if test:
            time.sleep(0.01)

    #display episode score
    if score > 0.0:
        print("Episode {} Score:{}".format(episode, score))
    #update log
    if total_score > highest:
        highest = total_score

    #don't change parameters if testing
    #display stats
    print("Total Score: {} Highest: {} Steps: {}".format(total_score, highest, step))
    total_score = 0.0

    #next episode
    episode += 1

env.close()
