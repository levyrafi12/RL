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
eps = 0.8
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
    #The Q-Table learning algorithm
    while j < 99:
        j += 1
        # TODO: Implement Q-Learning
        prob = np.random.uniform(0,1)

        # print("num actions %d" % (env.action_space.n))
        if prob < eps or j <= 0: # reflect the noise
            action = random.randint(0, env.action_space.n - 1)
        else:
            action = np.argmax(Q[state] + np.random.randn(1,4)) 
        env.render()
        # print("set action %d" % (action))
        next_state, reward, done, _ = env.step(action)

        Qmax = np.max([Qprev for Qprev in Q[next_state]])
        Q[state, action] = lr * (reward + y * Qmax) + (1 - lr) * Q[state, action]
        rAll += reward

        print("j=%d R=%d next_state=%d done=%d" % (j, reward, next_state, done))
        print("Q[%d, %d]=%f" % (state, action, Q[state, action]))

        state = next_state

        if done:
            break
        # 1. Choose an action by greedily (with noise) picking from Q table
        # 2. Get new state and reward from environment
        # 3. Update Q-Table with new knowledge
        # 4. Update total reward
        # 5. Update episode if we reached the Goal State

    rList.append(rAll)

# Reports
print("Score over time: " +  str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)