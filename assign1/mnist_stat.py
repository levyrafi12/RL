import matplotlib.pyplot as plt

def print_statistics(epoch_hist, loss_hist, epoch, i, running_loss):
	epoch_hist.append(epoch + (i / 600.0))
	loss_hist.append(running_loss / 200.0)
	print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
	return epoch_hist, loss_hist
