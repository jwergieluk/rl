import numpy
import time
import subprocess
import gym
import random


def render(env, cum_reward):
    subprocess.call('clear', shell=False)
    env.render()
    print(f'cum_rew {cum_reward}')
    time.sleep(1.0 / 5.0)


class Agent1:
    def __init__(self, env):
        self.dim_s = env.observation_space.n
        self.dim_a = env.action_space.n

    def step(self, state, action, reward, next_state, done):
        pass

    def select_action(self, state):
        return random.randint(0, self.dim_a-1)


def run(env, agent, max_episodes=10):
    state = env.reset()
    cum_reward = 0
    for _ in range(max_episodes):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        agent.step(state, action, reward, next_state, done)
        cum_reward += reward
        state = next_state
        if done:
            return cum_reward

        render(env, cum_reward)


if __name__ == '__main__':
    environment = gym.make('Taxi-v2')
    end_reward = run(environment, Agent1(environment))

