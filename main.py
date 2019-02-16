from rl01 import Agent1
from monitor import interact
import gym


env = gym.make('Taxi-v2')
agent = Agent1(env)
avg_rewards, best_avg_reward = interact(env, agent)