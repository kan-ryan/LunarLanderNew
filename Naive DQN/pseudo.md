# Lunar Landing (Vanilla Q-learning)

## Design

### Core

#### global methods
**run_episodes(num_episodes: int, agent: Any, do_training: bool) -> int:**
do the while loop on slide 13

**load_agent(filepath)**

**save_agent(filepath, agent)**


#### class Agent

**get_action(state)**

**update_q(state, action, next_state, reward)**


### Training 

**cmd inputs**
agent_filepath
gamma
epsilon
exploration_method: "g", "r", "e"

1. train agent
	- call run_episodes

2. save agent


### Testing

**cmd inputs**
agent_filepath

1. load agent

2. run agent
	- call run_episodes
	- mean/std of the rewards across episodes

3. save reward stats