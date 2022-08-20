import gym
env = gym.make("LunarLander-v2")
env.action_space.seed(42)

observation= env.reset()

for i in range(10000):
    env.render()
    for i in range(5000):
        observation, reward, done, info = env.step(env.action_space.sample())
    env.reset()
    

env.close()