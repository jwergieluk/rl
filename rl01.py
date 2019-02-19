import sys
import math
import numpy
import time
import subprocess
import gym
import random
import networkx


def render(env, i, cum_reward):
    subprocess.call('clear', shell=False)
    env.render()
    print(f'ep {i}  cum_reward {cum_reward}')
    time.sleep(1.0 / 2.0)


def encode(taxi_row, taxi_col, pass_loc, dest_idx):
    # (5) 5, 5, 4
    i = taxi_row
    i *= 5
    i += taxi_col
    i *= 5
    i += pass_loc
    i *= 4
    i += dest_idx
    return i


def decode(i):
    out = [i % 4]
    i = i // 4
    out.append(i % 5)
    i = i // 5
    out.append(i % 5)
    i = i // 5
    out.append(i)
    assert 0 <= i < 5
    return reversed(out)


class TaxiPolicyEval:
    """
    +---------+
    |R: | : :G|
    | : : : : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+ """

    edges = [((1, 2), (0, 2)), ((1, 2), (1, 1)), ((1, 2), (2, 2)), ((1, 2), (1, 3)), ((0, 2), (0, 3)), ((0, 3), (1, 3)), ((0, 3), (0, 4)), ((1, 3), (2, 3)), ((1, 3), (1, 4)), ((2, 3), (2, 2)), ((2, 3), (3, 3)), ((2, 3), (2, 4)), ((2, 2), (3, 2)), ((2, 2), (2, 1)), ((3, 2), (3, 1)), ((3, 2), (4, 2)), ((3, 1), (2, 1)), ((3, 1), (4, 1)), ((2, 1), (2, 0)), ((2, 1), (1, 1)), ((2, 0), (1, 0)), ((2, 0), (3, 0)), ((1, 0), (0, 0)), ((1, 0), (1, 1)), ((0, 0), (0, 1)), ((1, 1), (0, 1)), ((3, 0), (4, 0)), ((4, 2), (4, 1)), ((3, 3), (3, 4)), ((3, 3), (4, 3)), ((2, 4), (3, 4)), ((2, 4), (1, 4)), ((3, 4), (4, 4)), ((1, 4), (0, 4)), ((4, 4), (4, 3))]
    locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
    drive = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}
    pass_loc_decode = {0: 'R', 1: 'G', 2: 'Y', 3: 'B', 4: 'T'}
    dest_decode = ['R', 'G', 'Y', 'B']
    action_decode = {0: 'south', 1: 'north', 2: 'east', 3: 'west', 4: 'pickup', 5: 'dropoff'}

    def __init__(self):
        self.graph = networkx.Graph()
        self.graph.add_edges_from(self.edges)

    def action_is_optimal(self, state, action):
        r, c, pass_loc, dest_idx = decode(state)
        sys.stdout.write(f'state {state}: {r} {c} {self.pass_loc_decode[pass_loc]} {self.dest_decode[dest_idx]}  ')
        sys.stdout.write(f'action {action}: {self.action_decode[action]}  ')

        pass_in_taxi = pass_loc == 4
        goal_tile = self.locs[dest_idx] if pass_in_taxi else self.locs[pass_loc]
        if (r, c) == goal_tile:
            if pass_in_taxi:
                return action == 5
            return action == 4
        if action in (4, 5):
            return False

        r1 = r + self.drive[action][0]
        c1 = c + self.drive[action][1]
        for path in networkx.all_shortest_paths(self.graph, source=(r, c), target=goal_tile):
            # print(list(path))
            if (r1, c1) == path[1]:
                return True
        else:
            return False

    def distance_to_optimal(self, policy):
        distance = 0
        for s, a in enumerate(policy):
            if not self.action_is_optimal(s, a):
                distance += 1
                print('not opt')
            else:
                print('')
        return distance


class Agent1:
    def __init__(self, env, alpha: float, gamma: float, epsilon: float, update_method: str):
        self.dim_s = env.observation_space.n
        self.dim_a = env.action_space.n
        self.Q = numpy.random.randn(self.dim_s, self.dim_a) + 15.0
        self.alpha = alpha
        self.gamma = gamma
        self.episodes_no = 1.0
        self.epsilon0 = epsilon
        if not hasattr(self, update_method):
            raise ValueError(f'Update method {update_method} not implemented')
        self.update_method = getattr(self, update_method)
        self.test_phase = False
        self.graph = networkx.Graph()

    def step(self, state, action, reward, next_state, done):
        if done:
            self.episodes_no += 1.0
        if not self.test_phase:
            self.update_method(state, action, reward, next_state)

    def sarsa(self, state, action, reward, next_state):
        """ Sarsa on-policy Q-table update rule """
        next_action = self.select_action(next_state)
        self.Q[state, action] = self.Q[state, action]*(1.0 - self.alpha) + self.alpha*(
                reward + self.gamma*self.Q[next_state, next_action])

    def sarsamax(self, state, action, reward, next_state):
        """ Sarsamax aka Q-learning off-policy Q-table update rule """
        next_action = numpy.argmax(self.Q[next_state, :])
        self.Q[state, action] = self.Q[state, action]*(1.0 - self.alpha) + self.alpha*(
                reward + self.gamma*self.Q[next_state, next_action])

    def expected_sarsa(self, state, action, reward, next_state):
        """ Expected Sarsa on-policy Q-table update rule """
        next_action = numpy.argmax(self.Q[next_state, :])
        policy_vector_for_next_state = numpy.repeat(self.epsilon()/self.dim_a, self.dim_a)
        policy_vector_for_next_state[next_action] += 1.0 - self.epsilon()
        self.Q[state, action] = self.Q[state, action] * (1.0 - self.alpha) + self.alpha * (
                reward + self.gamma * numpy.dot(policy_vector_for_next_state, self.Q[next_state, :]))

    def epsilon(self):
        if self.test_phase:
            return 0.0
        n = self.episodes_no
        if n < 11000.0:
            return 0.1
        return self.epsilon0

    def select_action(self, state):
        if random.random() <= self.epsilon():
            return random.randint(0, self.dim_a-1)
        return numpy.argmax(self.Q[state, :])

    def get_policy(self):
        return numpy.argmax(self.Q, axis=1)


def run(env, agent, max_episodes=20000, verbose=True):
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
        if i % 500 == 0 and verbose:
            print(i, best_avg_reward, numpy.mean(agent.Q))
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
    ag = Agent1(environment, alpha=0.05, gamma=0.9, epsilon=0.0, update_method='expected_sarsa')
    run(environment, ag, max_episodes=20000)
    ag.test_phase = True

    mean_end_reward = run(environment, ag, max_episodes=10000, verbose=False)
    print('max_mean_reward', mean_end_reward)

    print(ag.get_policy())

    policy_eval = TaxiPolicyEval()
    print(policy_eval.distance_to_optimal(ag.get_policy()))



