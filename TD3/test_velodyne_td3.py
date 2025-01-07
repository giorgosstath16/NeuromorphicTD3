import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from velodyne_env import GazeboEnv

class NeuromorphicLayer(nn.Module):
    def __init__(self, input_dim, output_dim, spike_threshold=0.5):
        super(NeuromorphicLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.weights = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        self.spike_threshold = nn.Parameter(torch.ones(output_dim) * spike_threshold)
        self.membrane_decay = nn.Parameter(torch.ones(output_dim) * 0.9)
        
    def forward(self, x, prev_membrane=None):
        batch_size = x.size(0)
        
        if prev_membrane is None:
            prev_membrane = torch.zeros(batch_size, self.output_dim, device=x.device)
            
        current = F.linear(x, self.weights, self.bias)
        
        membrane_potential = (
            self.membrane_decay * prev_membrane + 
            (1 - self.membrane_decay).unsqueeze(0) * current
        )
        
        spikes = (membrane_potential > self.spike_threshold).float()
        membrane_potential = membrane_potential * (1 - spikes)
        output = membrane_potential * spikes
        
        return output, membrane_potential

class NeuromorphicActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(NeuromorphicActor, self).__init__()
        self.layer1 = NeuromorphicLayer(state_dim, hidden_dim)
        self.layer2 = NeuromorphicLayer(hidden_dim, action_dim)
        
    def forward(self, state):
        x, m1 = self.layer1(state)
        x = F.relu(x)
        actions, _ = self.layer2(x)
        return torch.tanh(actions)

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = NeuromorphicActor(state_dim, action_dim).to(self.device)
        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )

# Set the parameters for testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
max_ep = 500
file_name = "NeuromorphicTD3_velodyne"

# Create the testing environment
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)

# Set seeds
torch.manual_seed(seed)
np.random.seed(seed)

# Initialize dimensions
state_dim = environment_dim + robot_dim
action_dim = 2
max_action = 1

# Create the network
network = TD3(state_dim, action_dim, max_action)
try:
    network.load(file_name, "./pytorch_models")
    print("Successfully loaded the model!")
except:
    raise ValueError("Could not load the stored model parameters")

# Testing loop
done = False
episode_timesteps = 0
state = env.reset()
count_rand_actions = 0
random_action = []

print("Starting testing...")
while True:
    # Select action
    action = network.select_action(np.array(state))
    
    # Add small noise for exploration stability
    action = (action + np.random.normal(0, 0.1, size=action_dim)).clip(-max_action, max_action)
    
    # Random actions near obstacles (similar to training)
    if min(state[4:-8]) < 0.6 and np.random.uniform(0, 1) > 0.85 and count_rand_actions < 1:
        count_rand_actions = np.random.randint(8, 15)
        random_action = np.random.uniform(-1, 1, 2)
    
    if count_rand_actions > 0:
        count_rand_actions -= 1
        action = random_action
        action[0] = -1  # Full reverse when near obstacles
    
    # Scale action to environment range
    a_in = [(action[0] + 1) / 2, action[1]]
    
    # Execute action
    next_state, reward, done, target = env.step(a_in)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)

    # Reset on episode end
    if done:
        print(f"Episode finished after {episode_timesteps + 1} timesteps")
        state = env.reset()
        done = False
        episode_timesteps = 0
        count_rand_actions = 0
        random_action = []
    else:
        state = next_state
        episode_timesteps += 1