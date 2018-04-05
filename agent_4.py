import gym
from gym import spaces
import random
import numpy as np

env = gym.make('CartPole-v0')

def run_episode(w):
    score = 0
    t = 0
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
    return score

best_w = np.array(np.zeros(4))
best_score = 0

for i in range(10000):
    w = [random.uniform(-1,1) for i in range(4)]
    w = np.array(w)
    score = run_episode(w)
    if score > best_score:
        best_w = np.array(w)
        best_score = score

print("Score {} and best weights {}".format(best_score, best_w))

quit()