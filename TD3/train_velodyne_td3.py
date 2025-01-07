import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from velodyne_env import GazeboEnv

def evaluate(network, env, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        done = False
        while not done and count < 501:
            action = network.select_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, _ = env.step(a_in)
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    print("..............................................")
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    print("..............................................")
    return avg_reward

class NeuromorphicLayer(nn.Module):
    def __init__(self, input_dim, output_dim, spike_threshold=0.5):
        super(NeuromorphicLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize weights with proper scaling
        self.weights = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # Membrane potential parameters
        self.spike_threshold = nn.Parameter(torch.ones(output_dim) * spike_threshold)
        self.membrane_decay = nn.Parameter(torch.ones(output_dim) * 0.9)
        
    def forward(self, x, prev_membrane=None):
        batch_size = x.size(0)
        
        # Initialize membrane potential if None
        if prev_membrane is None:
            prev_membrane = torch.zeros(batch_size, self.output_dim, device=x.device)
            
        # Compute current input contribution
        current = F.linear(x, self.weights, self.bias)
        
        # Update membrane potential
        membrane_potential = (
            self.membrane_decay * prev_membrane + 
            (1 - self.membrane_decay).unsqueeze(0) * current
        )
        
        # Generate spikes
        spikes = (membrane_potential > self.spike_threshold).float()
        
        # Reset membrane potential where spikes occurred
        membrane_potential = membrane_potential * (1 - spikes)
        
        # Output weighted by spikes
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

class NeuromorphicCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(NeuromorphicCritic, self).__init__()
        # Q1 architecture
        self.layer1_1 = NeuromorphicLayer(state_dim + action_dim, hidden_dim)
        self.layer1_2 = NeuromorphicLayer(hidden_dim, 1)
        
        # Q2 architecture
        self.layer2_1 = NeuromorphicLayer(state_dim + action_dim, hidden_dim)
        self.layer2_2 = NeuromorphicLayer(hidden_dim, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        # Q1
        q1, m1 = self.layer1_1(sa)
        q1 = F.relu(q1)
        q1, _ = self.layer1_2(q1)
        
        # Q2
        q2, m2 = self.layer2_1(sa)
        q2 = F.relu(q2)
        q2, _ = self.layer2_2(q2)
        
        return q1, q2

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = NeuromorphicActor(state_dim, action_dim).to(self.device)
        self.actor_target = NeuromorphicActor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = NeuromorphicCritic(state_dim, action_dim).to(self.device)
        self.critic_target = NeuromorphicCritic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99,
             tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        
        for it in range(iterations):
            # Sample replay buffer
            batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states = \
                replay_buffer.sample_batch(batch_size)
            state = torch.FloatTensor(batch_states).to(self.device)
            action = torch.FloatTensor(batch_actions).to(self.device)
            reward = torch.FloatTensor(batch_rewards).to(self.device)
            done = torch.FloatTensor(batch_dones).to(self.device)
            next_state = torch.FloatTensor(batch_next_states).to(self.device)

            # Select next action according to target policy
            noise = torch.FloatTensor(batch_actions).data.normal_(0, policy_noise).to(self.device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute target Q-value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Optimize Critic
            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:
                # Compute actor loss
                actor_loss = -self.critic(state, self.actor(state))[0].mean()
                
                # Optimize actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft update target networks
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

# Training Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0  
eval_freq = 5e3
max_ep = 500
eval_ep = 10
max_timesteps = 5e6
expl_noise = 1
expl_decay_steps = 500000
expl_min = 0.1
batch_size = 40
discount = 0.99999
tau = 0.005
policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2
buffer_size = 1e6
file_name = "NeuromorphicTD3_velodyne"
save_model = True
load_model = False
random_near_obstacle = True

# Create directories
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

# Environment setup
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)

# Set random seeds
torch.manual_seed(seed)
np.random.seed(seed)

# Initialize network and buffer
state_dim = environment_dim + robot_dim
action_dim = 2
max_action = 1

network = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer(buffer_size, seed)

if load_model:
    try:
        network.load(file_name, "./pytorch_models")
        print("Loaded model successfully!")
    except:
        print("Could not load model")

# Training Loop
timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1

count_rand_actions = 0
random_action = []
evaluations = []

# Main training loop
while timestep < max_timesteps:
    if done:
        if timestep != 0:
            print(f"Total Timesteps: {timestep} Episode Num: {episode_num} Reward: {episode_reward:.3f}")
            network.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

        if timesteps_since_eval >= eval_freq:
            timesteps_since_eval %= eval_freq
            evaluations.append(evaluate(network, env, epoch, eval_ep))
            network.save(file_name, directory="./pytorch_models")
            np.save("./results/%s" % (file_name), evaluations)
            epoch += 1

        state = env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # Select action with exploration noise
    if expl_noise > expl_min:
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

    action = network.select_action(np.array(state))
    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(-max_action, max_action)

    # Random actions near obstacles
    if random_near_obstacle:
        if np.random.uniform(0, 1) > 0.85 and min(state[4:-8]) < 0.6 and count_rand_actions < 1:
            count_rand_actions = np.random.randint(8, 15)
            random_action = np.random.uniform(-1, 1, 2)

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action
            action[0] = -1

    # Execute action
    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)
    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    
    # Store transition
    replay_buffer.add(state, action, reward, done_bool, next_state)
    
    state = next_state
    episode_reward += reward
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# Final evaluation and save
evaluations.append(evaluate(network, env, epoch, eval_ep))
if save_model:
    network.save("%s" % file_name, directory="./pytorch_models")
np.save("./results/%s" % file_name, evaluations)