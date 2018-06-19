import pickle
import os.path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

step_size = 10000

plt.xlabel('Timestep')
plt.ylabel('reward')

def plot_data(data, color):
	time_steps = [i for i in range(len(data))]
	plt.plot(time_steps, data, color)

with open(os.path.join(".\\", "statistics.pkl"), "rb") as f:
	dict = pickle.load(f)
	plot_data(dict['best_mean_episode_rewards'], 'b')
	plot_data(dict['mean_episode_rewards'], 'r')

red_patch = mpatches.Patch(color='red', label='mean reward (100 episodes)')
blue_patch = mpatches.Patch(color='blue', label='best mean reward')

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.legend(handles=[red_patch, blue_patch], loc=4)

plt.title("Learning curve on Pong game using default hyperparameters")
# plt.show()

plt.savefig('learning_curve_default.png')



