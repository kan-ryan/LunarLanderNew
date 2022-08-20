from core import run_episodes
from core import save_agent
from core import load_agent
from QLearningAgent import QLearningAgent
import sys




if sys.argv[1] == "train": 
    agent = QLearningAgent(0.5, 0.5, 4)
    run_episodes(num_episodes = 3000, agent = agent, do_training = True, episode_time=6000, epsilon=0.75)
    save_agent(agent=agent, num_episodes=3000, episode_time=6000, epsilon=0.75)
else:

    agent = load_agent(sys.argv[2])
    run_episodes(agent=agent, num_episodes=20, episode_time=30000, do_training=False)


