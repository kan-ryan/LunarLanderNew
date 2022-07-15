import gym
import numpy as np
import pickle 
from QLearningAgent import QLearningAgent
import json
def run_episodes(num_episodes, agent, do_training, episode_time, epsilon=None):
    env = gym.make('LunarLander-v2')
    for i_episode in range(num_episodes):
        state = env.reset()
        state = digitize_state(state, 5)
        

        for t in range(episode_time):
            if not do_training:
                env.render()
            
            action = agent.get_epsilon_greedy_action(state, epsilon) if do_training else agent.get_best_action(state)

            next_state, reward, done, info = env.step(action)
            next_state = digitize_state(next_state, 5)

            if do_training:
                agent.update_q(state, action, next_state, reward, done) 
            if done:
                #print("Episode finished after {} timesteps".format(t+1))
                break
            state = next_state
    env.close()

def digitize_state(raw_state, nintervals):
    state = raw_state[0:6]

    

    bins = np.linspace(-1, 1, nintervals)
    new_bins = np.zeros(len(bins)+2)

    new_bins[1:-1] = bins
    new_bins[0] = -np.inf
    new_bins[-1] = np.inf

    midpoints = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
    new_midpoints = np.array([-1]+midpoints+[1])

    d = np.digitize(state, new_bins)
    #new_state = midpoints[d-2]

    new_state = new_midpoints[d-1]
    raw_state[0:6] = new_state
    return tuple(raw_state)

def save_agent(agent, num_episodes, episode_time, epsilon):
    results = dict()
    results["num_episodes"] = num_episodes
    results["episode_time"] = episode_time
    results["epsilon"] = epsilon
    results["q_values"] = agent.Q
    results["alpha"] = agent.alpha
    results["gamma"] = agent.gamma
    results["num_actions"] = len(agent.actions)

    

    with open(file = "results1", mode = 'wb') as f:
        pickle.dump(results, f)

def load_agent(filepath):

    with open(file = filepath, mode = 'rb') as f:
        d = pickle.load(f)
    print(d.get("num_actions"))
    agent = QLearningAgent(d.get("alpha"), d.get("gamma"), d.get("num_actions"))
    agent.Q = d.get("q_values")
    return agent


















