#!/Users/stephen/miniconda3/bin/python
import torch
import torchvision

import gym
import numpy
import sys
import math
import random
import keyboard
import cv2
import skimage.measure

from skimage.transform import resize
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'next_state', 'reward', 'qvalues'))

def preprocess(state):
    train = False
    if len(state) > 4:
        train = True
        state = state[:,:,10:,:,:]
        state = (torch.sum(state, dim=4) / 3.0)
    else:
        state = state[:,10:,:,:]
        state = (torch.sum(state, dim=3) / 3.0)

    if len(state.shape) == 3:
        state = state.unsqueeze(0)

    if not train and False:
        preview = torch.tensor(state).cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
        cv2.imshow("state", preview)
        cv2.waitKey(1)

    state = torch.tensor(state) / 255.0
    return state.float()


env = gym.make("CartPole-v0")

init_episode_frameskip = 0
episode_steps = 1000
env._max_episode_steps = episode_steps

state = env.reset()

for i in range(init_episode_frameskip):
    state, _, _, _ = env.step(0)

p_states = 3

#state = resize(state, (128, 128, 3))
states = [torch.tensor(state) for _ in range(p_states)]

# env.render()
BATCH_SIZE = 1000

rewards = []
actions = []
next_states = []
current_q_values = []

score = 0
highest = 1
highest_steps = 1
sum_loss = 0.0
update_counter = 0

hidden = 512

in_features = p_states * 4
epsilon = 0.5

frameskip = 1
rate = 0.0001
gamma = 0.5

n_actions = env.action_space.n
lives = 3

action = 0

render = False
first = True

replay_memory = []

def one_hot(x):
    a = torch.zeros(n_actions)
    a[x] = 1.0
    return a
"""
class Agent(torch.nn.Module):
    def __init__(self, rate, in_features):

        super(Agent, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(p_states, 32, 4, 4),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            #torch.nn.BatchNorm2d(16),

            torch.nn.Conv2d(32, 128, 4, 2),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            #torch.nn.BatchNorm2d(32),

            torch.nn.Conv2d(128, hidden, 2, 1),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
        )
        self.actora = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        x = preprocess(x)
        b = x.shape[0]
        x = self.conv(x)

        ax = self.actora(x.view(b, -1))
       # bx = self.actorb(x.view(b, -1))

        return ax.view(b, -1).cpu()#, bx.view(b, -1).cpu()
"""
class Agent(torch.nn.Module):
    def __init__(self, rate, in_features):

        super(Agent, self).__init__()
        self.actora = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_actions),
        )
        self.actorb = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        b = x.shape[0]
        ax = self.actora(x.view(b, -1).float())
        bx = self.actorb(x.view(b, -1).float())

        return torch.min(ax.view(b, -1).cpu(), bx.view(b, -1).cpu())

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier(m.weight.data, 0.0, 0.02)
        torch.nn.fill(m.weight.bias, 1.0)
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier(m.weight.data, 0.0, 0.02)
        torch.nn.fill(m.weight.bias, 1.0)

target_agent = Agent(rate, in_features).float()
agent = Agent(rate, in_features).float()
#weights_init(agent)
target_agent.load_state_dict(agent.state_dict())

optimizer = torch.optim.Adam(lr=rate, params=agent.parameters())

# try to load saved agent, if not found start new
try:
    checkpoint = torch.load("./agents/{}".format(env_name))
    agent.load_state_dict(checkpoint['model_state_dict'])
    train = False
    greedy = True
except:
    print("New agent.")


# p_agent.load_state_dict(agent.state_dict())

#weights_init(target_agent)

steps = 0
epoch = 0
q = torch.zeros(n_actions)
new_q = q
n_epochs = 5
games = 0

def discount(rewards):
    R = 0.0
    discount_rewards = []

    for r in reversed(rewards):
        R = r + (R * gamma)  # sum of discounted rewards
        discount_rewards.insert(0, R)

    discount_rewards = torch.tensor(discount_rewards)

    return discount_rewards

def learn(gamma):
    sys.stdout.flush()

    batch = random.sample(replay_memory, len(replay_memory))

    state_batch = torch.stack([b.state for b in batch])
    next_states_batch = torch.stack([b.next_state for b in batch]).squeeze(1)
    c_q = torch.stack([b.qvalues for b in batch])
    reward_batch = torch.tensor([b.reward for b in batch])

    next_state_valuesa = agent.forward(next_states_batch)
    # steps x q_values

    next_state_values = next_state_valuesa.squeeze(1).max(1)[0]

    # Compute the expected Q values'

    #   print(next_state_values.shape, discount_rewards)
    expected_state_action_values = (next_state_values * gamma) + reward_batch.unsqueeze(0).float()

    # Compute Huber loss
    c_q = c_q.squeeze(1).squeeze(1)

    loss = torch.nn.SmoothL1Loss()(c_q.max(1)[0], expected_state_action_values)
    l = loss.item()

    # Optimize the model
    optimizer.zero_grad()

    loss.backward(retain_graph=True)
    optimizer.step()

    del loss
    del batch
    del state_batch
    del reward_batch
    del expected_state_action_values
    del next_states_batch
    del c_q
    del next_state_values
    torch.cuda.empty_cache()
    # replay_memory.clear()

    return l

def learn_single(gamma, state, new_state, q, reward, done):
    # ("Training... Epoch",end="")

    next_state_values = target_agent.forward(new_state)
    # steps x q_values

    next_state_values = next_state_values.squeeze(1)
    next_state_values = next_state_values.max(1)[0]

    # Compute the expected Q values
    if not done:
        expected_state_action_values = (next_state_values * gamma) + torch.tensor(reward).float()
    else:
        expected_state_action_values = torch.tensor(reward).float()

    # Compute Huber loss
    q = q.squeeze(1).squeeze(1)

    loss = torch.nn.MSELoss()(q.max(1)[0], expected_state_action_values)
    l = loss.item()

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return l


total_reward = 0.0

def savemodel():
    a = input("Save? <Y/N>")
    if a == "Y" or a == "y":
        print("Saving agent parameters...")
        modelname = "./agents/{}".format(env_name)

        torch.save({
            'model_state_dict': agent.state_dict(),
        }, modelname)
    env.close()


# atexit.register(savemodel)
anim = ["|", "\\", "-", "/"]


def push_replay_memory(*args):
    replay_memory.append(Transition(*args))


reward = 0.0
confidence = 1.0
episode_games = 0
old_state = torch.stack(states).view(-1)


while True:

    if keyboard.is_pressed(" "):
        render = not render

    q = new_q

    state = torch.stack(states).view(-1)

    anxiety = torch.nn.Sigmoid()(torch.randn(1))

    r = torch.nn.Sigmoid()(torch.randn(1))
    if r < confidence-0.01:
        new_q = agent.forward(state.unsqueeze(0))
        action = torch.argmax(new_q, dim=-1)
        push_replay_memory(torch.tensor(old_state), torch.tensor(state), reward, new_q)
    else:
        action = env.action_space.sample()

    if render and steps % 1 == 0:
        env.render()

    steps += 1
    update_counter += 1

    old_state = state

    if first:
        action = 0
        first = False

    reward = 0.0
    for i in range(frameskip):
        new_state, r, done, x = env.step(int(action))
       # new_state = resize(new_state, (128, 128, 3))
        if r > 0:
            r = 1
        else:
            r = 0
        score += r
        reward += r
        if done:
            break

    states.append(torch.tensor(new_state))
    del states[0]

    # cv2.imshow("state", state.squeeze(0).permute(1,2,0).cpu().numpy())
    # cv2.waitKey(1)

    # print("\r{}".format(steps), end="")
    #sum_loss += learn_single(gamma, state, state, new_q, reward, done)
    #replay_memory.clear()

    state = new_state

    if done:
        games += 1
        #confidence = steps / (highest + 1)
        print(confidence)

        episode_games += 1

        if len(replay_memory) >= BATCH_SIZE:
            sum_loss = learn(gamma)
            replay_memory.clear()
            steps = 0
            score = 0
            episode_games = 0
            epoch += 1
            print("\n")

            if epoch % 10 == 0:
                update_counter += 1
                target_agent.load_state_dict(agent.state_dict())
                print("\nTarget net update.\n")

        first = True
        # for p in agent.parameters():
        #    print(p)
        print("\rGame/Epoch {}/{} | Steps {} | Avg Score {} | Highest: {} | Loss: {}".format(games, epoch, steps, int(score/(episode_games+1)), highest, sum_loss),end="")

        if steps > highest:
            highest = steps
            episode_steps += 100
            env._max_episode_steps = episode_steps

        # env.render()
        state = env.reset()

        for i in range(init_episode_frameskip):
            state, _, _, _ = env.step(0)

        #state = resize(state, (128, 128, 3))
        states = [torch.tensor(state) for _ in range(p_states)]

        action = torch.tensor([0]).float()
        p_action = torch.tensor([0]).float()

        #if score > highest:
        #    highest = score

        reward = 0.0
        sum_loss = 0.0
        total_reward = 0.0

        agent.hiddena = torch.zeros(1, 1, hidden)
