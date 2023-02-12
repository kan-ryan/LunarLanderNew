import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import gym

import random
import numpy as np



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque

#get environment
env = gym.make('LunarLander-v2')

#seeding enviornment to get repeatable results
env.seed(0)

class DQNModel(nn.Module):

    def __init__(self, state_size, action_size, seed):
        """
        state_size: dimension of the state
        action_size: dimension of each action
        seed: random seed
        """

        super(DQNModel, self).__init__()
        self.seed = torch.manual_seed(seed)

        # # # # # # # # # # # # # # # # # #
        # creating fully connected layers #
        # # # # # # # # # # # # # # # # # #

        #input layer that takes state_size
        self.fc1 = nn.Linear(state_size, 512)
        #hidden layer 1
        self.fc2 = nn.Linear(512, 256)
        #hidden layer 2
        self.fc3 = nn.Linear(256, 128)
        #hidden layer 3
        self.fc4 = nn.Linear(128, 64)
        #output layer that outputs action_size
        self.fc5 = nn.Linear(64, action_size)
    
    def forward(self, state):
        """ zeroes negative values in each tensor after each layer and returns the final tensor """
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)

        return self.fc5(x)



BUFFER_SIZE = int(1e5)  # size of replay buffer
BATCH_SIZE = 64         # size of batch
GAMMA = 0.99            # discount factor
TAU = 1e-3              # used for updating target net
LR = 5e-4               # learning rate
TARGET_UPDATE = 4       # frequency to update target net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayMemory:
    """Replay Memory to save trained observation"""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        action_size: size of each action
        buffer_size: max size of buffer
        batch_size: size of training batch
        seed: random see
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.transition = namedtuple("Transition", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Adding new transition to the replay memory"""
        trans = self.transition(state, action, reward, next_state, done)
        self.memory.append(trans)
    
    def sample(self):
        """Randomly sample a batch of transitions"""

        # Get a random sample of transitions
        transitions = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([t.state for t in transitions if t is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([t.action for t in transitions if t is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([t.reward for t in transitions if t is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([t.next_state for t in transitions if t is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([t.done for t in transitions if t is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """return size of memory"""
        return len(self.memory)

class Agent():
    

    def __init__(self, state_size, action_size, seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        #Two Neural Networks for Temporal Difference Learning

        #Policy Net
        self.policy_net = DQNModel(state_size, action_size, seed).to(device)
        #Target Net
        self.target_net = DQNModel(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)

       
        self.memory = ReplayMemory(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
       
        self.steps_done = 0
    
    def step(self, state, action, reward, next_state, done):
        #Take a step in the environment with a batch of transitions
        self.memory.add(state, action, reward, next_state, done)
        
       
        self.steps_done = (self.steps_done + 1) % TARGET_UPDATE
        if self.steps_done == 0:
            
            if len(self.memory) > BATCH_SIZE:
                transitions = self.memory.sample()
                self.optimize(transitions, GAMMA)

    def get_action(self, state, eps=0.):
        # Gets evaluate model and get an action 

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.policy_net.eval()
        with torch.no_grad():
            action_vals = self.policy_net(state)
        self.policy_net.train()

        # epsilon value check for epsilon greedy 
        if random.random() > eps:
            return np.argmax(action_vals.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def optimize(self, transitions, gamma):
        
        states, actions, rewards, next_states, dones = transitions

       
        next_target = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
       
        target = rewards + gamma * next_target * (1 - dones)
        
        expect = self.policy_net(states).gather(1, actions)
        
   
        loss = F.huber_loss(expect, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

       
        self.update(self.policy_net, self.target_net, TAU)                     

    def update(self, policyNN, targetNN, tau):
      
        for target_param, local_param in zip(targetNN.parameters(), policyNN.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)







def train(num_episodes=2000, max_timesteps=800, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    # training the agent and opening the environment on screen once a certain reward is reached
    scores = []                       
    scores_window = deque(maxlen=100) 
    eps = eps_start                
    show = False
    for i in range(1, num_episodes+1):
        state = env.reset()
        score = 0
    
        
        for j in range(max_timesteps):
            if show == True:    
                env.render()
            action = agent.get_action(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)      
        scores.append(score)
       
        print(np.mean(scores_window))             
        eps = max(eps_end, eps_decay*eps) 
        #set show env to True once a reward of 200 is reached
        if np.mean(scores_window) >= 200.0:
            print(i)
            print(j)
            show = True
        #close the enviornment after a reward greater than 215 is reached
        if np.mean(scores_window)>=215.0:
            env.close()
            break
        
    return scores


agent = Agent(state_size=8, action_size=4, seed=0)
scores = train()
