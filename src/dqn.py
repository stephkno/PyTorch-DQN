#!/usr/local/bin/python3
import gymnasium as gym
import ale_py
import torch
import random
from collections import namedtuple
from coach import Coach

# Register ALE environments
gym.register_envs(ale_py)

# ----------------- DQN Agent -----------------
class Agent(torch.nn.Module):
    def __init__(self, in_size=4, hidden=64, layers=1, out=2, device=None):
        super(Agent, self).__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        modules = [torch.nn.Linear(in_size, hidden), torch.nn.ReLU()]

        for _ in range(layers):
            modules.append(torch.nn.Linear(hidden, hidden))
            modules.append(torch.nn.ReLU())
        modules.append(torch.nn.Linear(hidden, out))

        self.actor = torch.nn.Sequential(*modules).to(self.device)
    def reset(self):
        pass

    def act(self, state):
        return self.actor(state)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

# ----------------- RNN DQN Agent -----------------
class RnnAgent(torch.nn.Module):
    def __init__(self, in_size=4, hidden=64, layers=1, out=2, device=None):
        super(RnnAgent, self).__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.gru = torch.nn.GRU(in_size, hidden, 4)
        
        modules = [torch.nn.Tanh(), torch.nn.Linear(hidden, out)]
        self.actor = torch.nn.Sequential(*modules).to(self.device)

        self.hidden = torch.zeros(4,1,256).cuda()

    def reset(self):
        self.hidden = torch.zeros(4,1,256).cuda()

    def act(self, state):
        if isinstance(state, torch.Tensor):
            state = state.to(self.device)
                
        state, self.hidden = self.gru(state, self.hidden)
        return self.actor(state)

    def init_weights(self):
        for m in self.actor.modules():
            if isinstance(m, torch.nn.Linear):
                # increase std for more variance in outputs
                torch.nn.init.normal_(m.weight, mean=0.0, std=1.0)  # larger spread
                torch.nn.init.uniform_(m.bias, -0.5, 0.5)



# ----------------- Preprocessing -----------------
def preprocess(state):
    """Convert environment state to torch tensor on GPU and normalize"""
    return torch.tensor(state).float().cuda()


# ----------------- Transition -----------------
transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


# ----------------- Agent Persistence -----------------
def save_agent(agent, optimizer, path, episode=None, step=None):
    """Save complete training state"""
    checkpoint = {
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode,
        'step': step,
    }
    
    # If you have a target network, save it too:
    # checkpoint['target_model_state_dict'] = target_model.state_dict()
    
    torch.save(checkpoint, path)
    print(f"Saved checkpoint at episode {episode}, step {step}")

def load_agent(agent, optimizer, path):
    """Load complete training state"""
    try:
        checkpoint = torch.load(path, map_location='cuda:0')
        
        agent.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # CRITICAL: Set to training mode
        agent.train()
        
        episode = checkpoint.get('episode', 0)
        step = checkpoint.get('step', 0)
        
        # If you have target network:
        # target_model.load_state_dict(checkpoint['target_model_state_dict'])
        
        print(f"Loaded checkpoint from episode {episode}, step {step}")
        return episode, step
        
    except FileNotFoundError:
        print(f"No saved agent found at {path}, starting fresh.")
        return 0, 0

# ----------------- Coach Wrapper -----------------
def make_coach(reward_shaping=None):
    return Coach(reward_shaping=reward_shaping, transition=Transition)


# ----------------- CUDA / Device Info -----------------
def print_device_info():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("CUDA not available - using CPU")
    print("-" * 50)
