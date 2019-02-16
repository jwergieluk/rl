import numpy
import time
import subprocess
import gym


def render(env, cum_reward):
    subprocess.call('clear', shell=False)
    env.render()
    print(cum_reward)
    time.sleep(1.0 / 5.0)


def run(env, max_episodes=10):
    env.reset()
    cum_reward = 0
    for _ in range(max_episodes):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        cum_reward += reward
        if done:
            return cum_reward

        render(env, cum_reward)


if __name__ == '__main__':
    environment = gym.make('Taxi-v2')
    end_reward = run(environment)

