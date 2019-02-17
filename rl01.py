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
        self.Q = numpy.random.randn(self.dim_s, self.dim_a) + 15.0
        self.alpha = 0.05
        self.gamma = 0.9
        self.episodes_no = 1.0

    def set_optimal(self):
        self.episodes_no = 20000.0

    def step(self, state, action, reward, next_state, done):
        if done:
            self.episodes_no += 1.0
        self.expected_sarsa_update(state, action, reward, next_state)

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
        policy_vector_for_next_state = numpy.repeat(self.epsilon()/self.dim_a, self.dim_a)
        policy_vector_for_next_state[next_action] += 1.0 - self.epsilon()
        self.Q[state, action] = self.Q[state, action] * (1.0 - self.alpha) + self.alpha * (
                reward + self.gamma * numpy.dot(policy_vector_for_next_state, self.Q[next_state, :]))

    def epsilon(self):
        n = self.episodes_no
        if n < 2500.0:
            return 0.1
#        if n < 5000.0:
#            return 0.7
#        if n < 10000.0:
#            return 0.5
#        if n < 15000.0:
#            return 0.2
#        if n < 19900.0:
#            return 0.01
        return 0.0

    def select_action(self, state):
        if random.random() <= self.epsilon():
            return random.randint(0, self.dim_a-1)
        return numpy.argmax(self.Q[state, :])


def train(env, agent, max_episodes=20000):
    end_rewards = []
    best_avg_reward = -math.inf
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
        avg = numpy.mean(end_rewards[-100:])
        best_avg_reward = avg if avg > best_avg_reward else best_avg_reward
        if i % 250 == 0:
            print(i, best_avg_reward, numpy.mean(agent.Q))
    return best_avg_reward


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
    max_mean_reward = train(environment, ag)
    print('max_mean_reward', max_mean_reward)

    #ag.set_optimal()
    #max_mean_reward = train(environment, ag, 100)
    # visualize(environment, ag)

