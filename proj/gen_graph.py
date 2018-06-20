import pickle
import os.path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

step_size = 10000

def init_plot():
	plt.close()
	plt.xlabel('Timestep')
	plt.ylabel('reward')
	plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	plt.grid(linestyle='dashed')
	

def plot_data(data, color, annotate):
	time_steps = [i for i in range(len(data))]
	plt.plot(time_steps, data, color)
	if annotate:
		annotate_max(time_steps, data)

def annotate_max(x_list, y_list):
	x = np.array(x_list)
	y = np.array(y_list)
	xmax = x[np.argmax(y)]
	ymax = y.max()
	text = "max y={:.3f}".format(ymax)
	ax = plt.gca()
	bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
	arrowprops = dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
	kw = dict(xycoords='data',textcoords="axes fraction",
		arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
	ax.annotate(text, xy=(xmax, ymax), xytext=(0.74,0.96), **kw)

def q1_gen_graph(stat_fn, title):
	init_plot()
	with open(os.path.join(".\\", stat_fn), "rb") as f:
		dict = pickle.load(f)
		plot_data(dict['best_mean_episode_rewards'], 'blue', True)
		plot_data(dict['mean_episode_rewards'], 'red', False)

	red_patch = mpatches.Patch(color='red', label='mean reward (100 episodes)')
	blue_patch = mpatches.Patch(color='blue', label='best mean reward')
	plt.legend(handles=[red_patch, blue_patch], loc=4)
	plt.title(title)

	start, end = plt.gca().get_ylim()
	plt.yticks(np.arange(-22.0, end + 5, 2))

	# plt.savefig('learn_curve_def_params.png')
	plt.show()

def q2_gen_graph(stat_files, colors, key, labels, title, out_pref, annotate):
	init_plot()
	color_patches = []
	for i in range(len(stat_files)):
		with open(os.path.join(".\\", stat_files[i]), "rb") as f:
			dict = pickle.load(f)
			plot_data(dict[key], colors[i], annotate[i])
			color_patches.append(mpatches.Patch(color=colors[i], label=labels[i]))
	
	plt.legend(handles=color_patches, loc=4)
	plt.title(title)
	# plt.savefig(out_pref + '_modified_params.png')
	plt.show()

if __name__ == '__main__':
	q1_gen_graph('statistics_def.pkl', 'mean and best mean rewards using default params')

	"""
	files = ['statistics_def.pkl', 'statistics_005.pkl', 'statistics_00.pkl']
	colors = ['red', 'blue', 'green']
	labels = ['final_p = 0.1', 'final_p = 0.05', 'final_p=0.0']
	annotate = [False, False, True]

	q2_gen_graph(files, colors, 'mean_episode_rewards', labels, 
		'mean reward using different final_p', 'mean_rew', annotate) 

	q2_gen_graph(files, colors, 'best_mean_episode_rewards', labels, 
		'best mean reward using different final_p', 'best_mean_rew', annotate) 
	"""





