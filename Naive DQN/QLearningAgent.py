import gym
from collections import defaultdict 
import numpy as np


class QLearningAgent():

	def __init__(self, alpha, gamma, num_actions):
		self.alpha = alpha
		self.gamma = gamma
		self.Q = defaultdict(float) #(state, action) -> float
		self.actions = list(range(num_actions))
		
		

		

	def get_epsilon_greedy_action(self, state, epsilon):
		rand = np.random.random_sample()
		return self.get_best_action(state) if rand > epsilon else np.random.choice(self.actions)


	def get_best_action(self, state):
		qVals = []
		for a in self.actions:
			qVals.append(self.Q[(state, a)])

		index = np.argmax(qVals)
		return self.actions[index] 



	def update_q(self, state, action, next_state, reward, terminal_state):
		
		maxQ = -float('inf')

		for a in self.actions:
			maxQ = max(self.Q[(next_state, a)], maxQ)
		if terminal_state:
			new_q = self.Q[(next_state, action)] + (self.alpha)*(reward - self.Q[(next_state, action)])
		else:
			new_q = (1-self.alpha)*self.Q[(state, action)] + (self.alpha)*(reward+self.gamma*maxQ)

		self.Q[(next_state, action)] = new_q

		
		



