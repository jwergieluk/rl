from collections import deque

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
        self.Q = numpy.zeros((self.dim_s, self.dim_a))
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.05

    def set_optimal(self):
        self.epsilon = 0.0

    def step(self, state, action, reward, next_state, done):
        self.sarsa_update(state, action, reward, next_state)

    def sarsa_update(self, state, action, reward, next_state):
        next_action = numpy.argmax(self.Q[next_state, :])
        self.Q[state, action] = self.Q[state, action]*(1.0 - self.alpha) + self.alpha*(
                reward + self.gamma*self.Q[next_state, next_action])

    def select_action(self, state):
        if random.random() <= self.epsilon:
            return random.randint(0, self.dim_a-1)
        return numpy.argmax(self.Q[state, :])


def train(env, agent, max_episodes=25000):
    end_rewards = deque(maxlen=100)

    for i in range(max_episodes):
        state = env.reset()
        cum_episode_reward = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            cum_episode_reward += reward
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        end_rewards.append(cum_episode_reward)
        mean_end_reward = numpy.mean(end_rewards)
        if i % 100 == 0:
            print(i, mean_end_reward)


def visualize(env, agent):
    state = env.reset()
    cum_episode_reward = 0
    while True:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        cum_episode_reward += reward
        render(env, cum_episode_reward)
        state = next_state
        if done:
            break


if __name__ == '__main__':
    environment = gym.make('Taxi-v2')
    ag = Agent1(environment)
    train(environment, ag)
    visualize(environment, ag)


