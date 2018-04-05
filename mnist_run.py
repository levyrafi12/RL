import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import mnist_1 
import mnist_2
import mnist_3

epoch_hist1, loss_hist1 = mnist_1.basic_config()
epoch_hist2, loss_hist2 = mnist_2.optimized_config()
epoch_hist3, loss_hist3 = mnist_3.hidden_layer_config()

plt.plot(epoch_hist1, loss_hist1, 'r', epoch_hist2, 
    loss_hist2, 'b', epoch_hist3, loss_hist3, 'g')

red_patch = mpatches.Patch(color='red', label='basic config')
blue_patch = mpatches.Patch(color='blue', label='optimized config')
green_patch = mpatches.Patch(color='green', label='hidden layer config')

plt.legend(handles=[red_patch, blue_patch, green_patch], loc=1)

plt.xlabel('epoch')
plt.ylabel('loss')
# plt.show()

plt.savefig('mnist_loss_curves.png')

quit()

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

plt.plot(t1, f(t1), 'r', t2, np.cos(2*np.pi*t2), 'b')

red_patch = mpatches.Patch(color='red', label='basic config')
blue_patch = mpatches.Patch(color='blue', label='optimized config')

plt.legend(handles=[red_patch, blue_patch], loc=1)

