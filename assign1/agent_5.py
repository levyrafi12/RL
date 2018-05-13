import gym
from gym import spaces
import random
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

w = np.array([0.54769932, -0.55488378, 0.86630044, 0.61971392])

def run_episodes():
    i_episode = 0
    score = 0
    while(score < 200):
        i_episode += 1
        t = 0
        score = 0
        observation = env.reset()
        while (True):
            t = t + 1
            env.render()
            prod = w.dot(observation)
            # pushing the cart to the right (w.o >= 0, action = 1)
            # pushing the cart to the left (w.o < 0, action = 0)
            action = 1 if prod >= 0 else 0
            observation, reward, done, info = env.step(action)
            score += reward
            if done:
                print("Episode finished after {} timesteps".format(t))
                break
    return i_episode

n_iter = 1000
total_num_episodes = 0
episodes_numbers = []

for i in range(n_iter):
    n_episode = run_episodes()
    total_num_episodes += n_episode
    episodes_numbers.append(n_episode)
average_num_episodes = total_num_episodes / n_iter

print("Average number of episodes reached a score of "
     "200 is %.5f" % (average_num_episodes))

plt.hist(episodes_numbers)
plt.ylabel("Frequency")
plt.xlabel("Episodes")
plt.title("Histogram Number of episodes reached score of 200")
# plt.show()

# plt.savefig('histogram_episodes.png')
quit()