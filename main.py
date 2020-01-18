import gym
import torch
import sys
from collections import namedtuple
import random

env = gym.make("MsPacman-ram-v0")

if len(sys.argv) > 1 and sys.argv[1].lower() == "test":
    test = True
else:
    test = False

#### Deep Q Network
# seemingly should converge after updating over a few million game steps
#
#
# ? ? ?
# manual seed for random initial weight generation
torch.manual_seed(0)

#hyperparameters
epoch = 0
epochs = 1000
episode = 1
init_action = 1
GAMMA = 0.999
rate = 0.005
TARGET_INTERVAL = 25
UPDATE_INTERVAL = 25
batch_size = 100

#define neural network model
class Agent(torch.nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
	        torch.nn.Tanh(),
            torch.nn.Linear(128, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 32),
            torch.nn.Tanh(),
        )
        self.head = torch.nn.Linear(32, env.action_space.n)

    def forward(self, x):
        y = self.model(x/255)
        return self.head(y)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
#replay memory for episodes
class Memory():
    def __init__(self, batch_size):
        super(Memory, self).__init__()
        self.buffer = []
        self.reset()
        self.batch_size = batch_size
        self.cap = 30000
        self.n = 150

    def push(self, *args):
        self.buffer.append(Transition(*args))
        if len(self.buffer) > self.cap:
            del self.buffer[0]
        else:
            self.length += 1

    def sample(self):
        return random.sample(self.buffer, self.batch_size)

    def reset(self):
        self.buffer = []
        self.length = 0
        self.total_loss = 0

    def learn(self):
        #only update if there are enough samples
        if memory.length >= memory.batch_size:
            print("\nUpdate Parameters")

            #run n epochs
            for e in range(self.n):
                #random sample experiences
                state, action, reward, next_state, dones = zip(*memory.sample())
                dones = torch.tensor(dones)*1.0

                #get max q values for 'next state' from new agent policy
                next_state = torch.stack(next_state)
                next_q_values = agent.forward(next_state).detach().max(1)[0]
                #zero out last step rewards
                next_q_values = next_q_values * dones
                #target value is what we want the neural network to predict
                targets = torch.tensor(reward) + (GAMMA * next_q_values)

                #get q values for 'this state' from old agent policy
                state = torch.stack(state)
                values = agent.forward(state)
                #get each q value for each action from sampled experiences
                values = torch.gather(values, dim=1, index=torch.tensor(action).unsqueeze(1)).squeeze(1)

                #run mean squared error against q targets and predicted q values
                loss = -1*(targets - values).pow(2).mean()
                print("{} - Loss:{}".format(e,loss.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

memory = Memory(batch_size)
agent = Agent()
target_agent = Agent()
print(agent)

if test:
    print("Load agent")
    agent.load_state_dict(torch.load("./checkpoint.pth"))

#define optimizer with learning rate (gradient step size)
optimizer = torch.optim.Adagrad(params=target_agent.parameters(), lr=rate)

def save_model(model):
    print("Saving model.")
    torch.save(model.state_dict(), './checkpoint.pth')

step = 0
highest = 0
print("Initializing replay buffer...")

#run forever
while True:
    done = False

    score = 0.0
    total_score = 0.0
    confidence = 1.0

    #reset game
    state = env.reset()
    env._max_episode_steps = 1000
    render = test

    if test:
        print("Loading newest agent")
        agent.load_state_dict(torch.load("./checkpoint.pth"))

    #initial step to get lives info
    _, _, _, info = env.step(init_action)
    if 'ale.lives' in info:
        lives = info["ale.lives"]
    state = torch.tensor(state).float()
    o_state = state

    #run episode
    while not done:

        #get Q values
        if test:
            target_agent.eval()
        q_values = agent.forward(state)

        #random number [0,1]
        r = torch.rand(1)

        #if random value higher than entropy value
        if r < confidence or test:
            #choose greedy action (exploit)
            action = torch.argmax(q_values)
        else:
            #choose random action (explore)
            action = env.action_space.sample()

        #take step in environment
        next_state, reward, done, info = env.step(action)
        next_state = torch.tensor(next_state).float()

        confidence *= 0.95
        confidence += reward
        if confidence < 0.001: confidence = 0.0
        confidence = max(min(confidence, 1.0), 0.2)

        #check if lost life (only for atari games)
        if 'ale.lives' in info and info["ale.lives"] < lives:
            lives = info["ale.lives"]
            reward = -1.0
        else:
            score += reward
            total_score += reward

        #render env for observing agent
        if render:
            env.render()

        #prime next state
        state = next_state

        #push this step experience to replay memory
        memory.push(state, action, reward, next_state, (not done))
        step += 1

    #display episode score
    if score > 0.0:
        print("Episode {} Score:{}".format(episode, score))
    #update log
    if total_score > highest:
        highest = total_score

    #don't change parameters if testing
    if not test:
        if episode % UPDATE_INTERVAL == 0:
            #update parameters
            memory.learn()
            render = not render

        if episode % TARGET_INTERVAL == 0:
            #update agent parameters
            print("Swap parameters")
            agent.load_state_dict(target_agent.state_dict())
            save_model(agent)

        #display stats
        if total_score > 0.0:
            print("Total Score: {} Highest: {} Steps: {}".format(total_score, highest, step))
        total_score = 0.0

    #next episode
    episode += 1

env.close()
