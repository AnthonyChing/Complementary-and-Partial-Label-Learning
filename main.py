# %%

# For debugging with Jupyter Notebook

import sys
sys.argv = [
    'main.py',  # dummy script name
    '--ds', 'clcifar10',
    '--algo', 'mcl',
    '--model', 'resnet18',
]

import os
import pickle
import gdown
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import argparse

# Import custom modules

from dataset import get_dataset
from algorithm import mcl_exp_loss, fwd_loss, evaluate_model
from model import get_resnet_model

# Parse command line arguments

parser = argparse.ArgumentParser(description="Complementary and Partial Label Learning")
parser.add_argument('--ds', type=str, required=True, choices=['clcifar10', 'clcifar20', 'clmin10', 'clmin20', 'cifar10_pl', 'cifar20_pl', 'min10_pl', 'min20_pl'], help='Dataset to use')
parser.add_argument('--algo', type=str, required=True, choices=['fwd', 'mcl', 'proden', 'pico'], help='Algorithm to use')
parser.add_argument('--model', type=str, required=True, choices=['resnet18', 'convnet'], help='Model architecture to use')
parser.add_argument('--bs', type=int, default=256, help='Batch size (default: 256)')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs (default: 300)')
args = parser.parse_args()

# Configure dataset

trainset, testset, num_classes = get_dataset(args)

# Configure model

model = get_resnet_model(num_classes)

# Configure algorithm

if args.algo == 'fwd':
    loss_function = fwd_loss
elif args.algo == 'mcl':
    loss_function = mcl_exp_loss
# Add more algorithms as needed
else:
    raise NotImplementedError(f"Algorithm {args.algo} not implemented yet.")

def main():
	trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4)
	testloader = DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	# Configure device
	 
	if torch.cuda.is_available():
		print("Using CUDA backend")
		device = torch.device("cuda")
	elif torch.backends.mps.is_available():
		print("Using MPS backend")
		device = torch.device("mps")
	else:
		print("Using CPU backend")
		device = torch.device("cpu")
	model.to(device)

	# Run training
	 
	num_epochs = args.epochs
	for epoch in range(num_epochs):
		
		model.train()
		running_loss = 0.0
		num_images = 0
		for images, cl_labels in trainloader:
			images = images.to(device)
			cl_labels = cl_labels.to(device)
			outputs = model(images)
			loss = loss_function(outputs, cl_labels)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			running_loss += loss.item() * images.size(0)
			num_images += images.size(0)

		# Evaluate current model on training set and test set
		
		# if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
		if True:
			model.eval()
			train_acc = evaluate_model(model, trainloader, device)
			test_acc = evaluate_model(model, testloader, device)
			print(f"Epoch {epoch+1}/{num_epochs}, Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%, Loss: {running_loss / num_images:.4f}")

if __name__ == '__main__':
    main()