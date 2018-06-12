"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import sys

from utils.replay_buffer import ReplayBuffer
from utils.gym import get_wrapper_by_name

USE_CUDA = torch.cuda.is_available()
# USE_CUDA = False

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# dtype = torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

Statistic = {
    "mean_episode_rewards": [],
    "best_mean_episode_rewards": []
}

def dqn_learing(
    env,
    q_func,
    optimizer_spec,
    exploration,
    stopping_criterion=None,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000
    ):

    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            input_channel: int
                number of channel of input.
            num_actions: int
                number of actions
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    exploration: Schedule (defined in utils.schedule)
        schedule for probability of chosing random action.
    stopping_criterion: (env) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_arg = env.observation_space.shape[0]
        print("# actions {}".format(env.action_space))
        print("input_arg {}".format(input_arg))
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_arg = frame_history_len * img_c
        print("# actions {}".format(env.action_space))
        print("input_arg {}".format(input_arg))
    num_actions = env.action_space.n

    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(model, obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
            with torch.no_grad():
                out = model(obs)
            return out.max(1)[1] 
        else:
            return torch.IntTensor([[random.randrange(num_actions)]])

    def evaluate_model(model, obs):
        if len(env.observation_space.shape) > 1:
            obs = torch.from_numpy(obs).type(dtype) / 255.0
        else:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
        return model(obs)

    # Initialize target q function and q function, i.e. build the model.
    Q_net = q_func(input_arg , num_actions).cuda() if torch.cuda.is_available() else q_func(input_arg , num_actions)
    Q_target_net = q_func(input_arg , num_actions).cuda() if torch.cuda.is_available() else q_func(input_arg , num_actions)

    # Construct Q network optimizer function
    optimizer = optimizer_spec.constructor(Q_net.parameters(), **optimizer_spec.kwargs)

    # Construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000
    # LOG_EVERY_N_STEPS = 10
    # learning_starts = 1

    stopping_criterion = None

    for t in count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env):
            break
       
        if t % 1000 == 0:
            print("t {}".format(t))

        frame_idx = replay_buffer.store_frame(last_obs)
        encoded_obs = replay_buffer.encode_recent_observation()
        action = select_epilson_greedy_action(Q_net, encoded_obs, t)
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.store_effect(frame_idx, action, reward, done)
        # env.render()
        # print("last_obs shape {}".format(last_obs.shape))
        if done:
            last_obs = env.reset()
            # print("episode completed after {} iterations , reward = {}".format(i, reward))
        else:
            last_obs = next_obs

        if (t > learning_starts and 
            t % learning_freq == 0 and 
            replay_buffer.can_sample(batch_size)):
            # print("learning..")
            obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = \
                replay_buffer.sample(batch_size)

            act_batch= torch.cat([torch.tensor([[a]], dtype=torch.long, \
                device=device) for a in act_batch])
            Q_val = evaluate_model(Q_net, obs_batch)
            # construct estimated Q - select Q[state(i), action(i)] for each i in batch
            Q_val = Q_val.squeeze(0).gather(1, act_batch).squeeze(1) 
            # print("Q_val {}".format(Q_val))
            # construct expected Q
            Q_next_max = evaluate_model(Q_target_net, next_obs_batch)
            # print("Q_next_max {}".format(Q_next_max))
            Q_next_max = Q_next_max.squeeze(0).max(1)[0]
            # print("Q_next_max.max(1)[0] {}".format(Q_next_max))
            y = torch.from_numpy(rew_batch).type(dtype) # expected Q
            # print("y {}".format(y))
            d_error = torch.tensor(torch.zeros(batch_size, device=device)) # Bellman delta error
            # print("d_error {}".format(d_error))
            for i in range(batch_size):
                y[i] += 0.0 if done_batch[i] else gamma * Q_next_max[i]
                # print("y[i] {} {}".format(i, y[i]))
                d_error[i] = -1 * (y[i] - Q_val[i]).clamp(-1, 1) 
            # print("d_error {}".format(d_error))
            # print("d_error.data.unsqueeze(1) {}".format(d_error.data.unsqueeze(1)))

            optimizer.zero_grad()
            current = Q_val.unsqueeze(1)
            # print("current {}".format(current))

            current.backward(d_error.data.unsqueeze(1))

            optimizer.step()

            if num_param_updates % target_update_freq == 0:
                Q_target_net.load_state_dict(Q_net.state_dict())  
                num_param_updates += 1

        ### 4. Log progress and keep track of statistics
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

        Statistic["mean_episode_rewards"].append(mean_episode_reward)
        Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)

        if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            sys.stdout.flush()

            # Dump statistics to pickle
            with open('statistics.pkl', 'wb') as f:
                pickle.dump(Statistic, f)
                print("Saved to %s" % 'statistics.pkl')
