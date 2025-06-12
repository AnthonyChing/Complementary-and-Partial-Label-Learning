import matplotlib.pyplot as plt
import re

def plot_log(filename):
	epochs = []
	accs = []
	losses = []
	with open(filename, 'r') as f:
		for line in f:
			match = re.match(r"Epoch (\d+).*Test Accuracy: ([\d.]+)%, Loss: ([\d.]+)", line)
			if match:
				epochs.append(int(match.group(1)))
				accs.append(float(match.group(2)))
				losses.append(float(match.group(3)))
	fig, ax1 = plt.subplots()
	color = 'tab:blue'
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Test Accuracy (%)', color=color)
	ax1.plot(epochs, accs, color=color, label='Test Accuracy')
	ax1.tick_params(axis='y', labelcolor=color)
	ax2 = ax1.twinx()
	color = 'tab:red'
	ax2.set_ylabel('Loss', color=color)
	ax2.plot(epochs, losses, color=color, label='Loss')
	ax2.tick_params(axis='y', labelcolor=color)
	
	# Mark the point with highest accuracy
	if accs:
		max_idx = max(range(len(accs)), key=lambda i: accs[i])
		ax1.scatter(epochs[max_idx], accs[max_idx], color='green', zorder=5)
		ax1.annotate(f'Max: {accs[max_idx]:.2f}%',
					 (epochs[max_idx], accs[max_idx]),
					 textcoords="offset points", xytext=(0,5), ha='center', color='green', fontsize=9, fontweight='bold')

	# Add grid lines
	ax1.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
	ax1.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5, alpha=0.7)

	plt.title('Proden on CIFAR10 (PL): Test Accuracy and Loss over Epochs')
	fig.tight_layout()
	plt.savefig('plcifar10_proden.png', dpi=300)
	plt.show()

if __name__ == '__main__':
	plot_log('plcifar10_proden.txt')
