import math
from collections import deque
import numpy
import time
import subprocess
import gym
import random


def render(env, i, cum_reward):
    subprocess.call('clear', shell=False)
    env.render()
    print(f'ep {i}  cum_reward {cum_reward}')
    time.sleep(1.0 / 2.0)


class Agent1:
    def __init__(self, env):
        self.dim_s = env.observation_space.n
        self.dim_a = env.action_space.n
        self.Q = numpy.zeros((self.dim_s, self.dim_a))
        self.alpha = 0.1
        self.gamma = 0.98
        self.epsilon = 0.3
        self.steps_no = 1.0

    def set_optimal(self):
        self.epsilon = 0.0

    def step(self, state, action, reward, next_state, done):
        self.steps_no += 1.0
        self.sarsa_update(state, action, reward, next_state)

    def sarsa_update(self, state, action, reward, next_state):
        """ Sarsa on-policy Q-table update rule """
        next_action = self.select_action(next_state)
        self.Q[state, action] = self.Q[state, action]*(1.0 - self.alpha) + self.alpha*(
                reward + self.gamma*self.Q[next_state, next_action])

    def sarsamax_update(self, state, action, reward, next_state):
        """ Sarsamax aka Q-learning off-policy Q-table update rule """
        next_action = numpy.argmax(self.Q[next_state, :])
        self.Q[state, action] = self.Q[state, action]*(1.0 - self.alpha) + self.alpha*(
                reward + self.gamma*self.Q[next_state, next_action])

    def expected_sarsa_update(self, state, action, reward, next_state):
        """ Expected Sarsa on-policy Q-table update rule """
        next_action = numpy.argmax(self.Q[next_state, :])
        policy_vector_for_next_state = numpy.repeat(self.epsilon/self.dim_a, self.dim_a)
        policy_vector_for_next_state[next_action] += 1.0 - self.epsilon
        self.Q[state, action] = self.Q[state, action] * (1.0 - self.alpha) + self.alpha * (
                reward + self.gamma * numpy.dot(policy_vector_for_next_state, self.Q[next_state, :]))

    def select_action(self, state):
        scale_factor = math.exp(-self.steps_no / 150000.0)
        if self.steps_no % 5000 == 0:
            print(self.steps_no, scale_factor)
        if random.random() <= self.epsilon * scale_factor:
            return random.randint(0, self.dim_a-1)
        return numpy.argmax(self.Q[state, :])


def train(env, agent, max_episodes=50000):
    end_rewards = []
    for i in range(max_episodes):
        state = env.reset()
        cum_reward = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            cum_reward += reward
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        end_rewards.append(cum_reward)
        if i % 500 == 0:
            print(i, cum_reward)
    return numpy.mean(end_rewards)


def visualize(env, agent):
    state = env.reset()
    cum_episode_reward = 0
    for i in range(1000):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        cum_episode_reward += reward
        render(env, i, cum_episode_reward)
        state = next_state
        if done:
            break


if __name__ == '__main__':
    environment = gym.make('Taxi-v2')
    ag = Agent1(environment)
    train(environment, ag)
    ag.set_optimal()
    max_mean_reward = train(environment, ag, 100)
    # visualize(environment, ag)

    print('max_mean_reward', max_mean_reward)

