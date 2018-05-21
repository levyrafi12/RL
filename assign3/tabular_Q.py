import gym
import numpy as np
import random

# Load environment
env = gym.make('FrozenLake-v0')

# Implement Q-Table learning algorithm
#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .8
y = .95
num_episodes = 2000
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    state = env.reset()
    rAll = 0 # Total reward during current episode
    d = False
    j = 0
    eps = 1.0 / (i + 1)
    #The Q-Table learning algorithm
    while j < 99:
        j += 1
        # TODO: Implement Q-Learning
        action = np.argmax(Q[state] + np.random.randn(1,env.action_space.n) * eps)
        next_state, reward, d, _ = env.step(action)
        Qmax = np.max(Q[next_state])
        Q[state, action] = lr * (reward + y * Qmax) + (1.0 - lr) * Q[state, action]
        rAll += reward
        state = next_state

        if d:
            break

    rList.append(rAll)

# Reports
print("Score over time: " +  str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)