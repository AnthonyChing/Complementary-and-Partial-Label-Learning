import os
import pickle
import numpy as np
import gdown
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torchvision
from PIL import Image
import torch.nn.functional as F

# Define the CLCIFAR10 dataset class
class CLCIFAR10(Dataset):
	def __init__(self, data, transform=None, num_classes=10):
		self.data = data
		self.transform = transform
		self.num_classes = num_classes

	def __len__(self):
		return len(self.data['images'])

	def __getitem__(self, idx):
		img, cl_label = self.data['images'][idx], self.data['cl_labels'][idx]
		img = Image.fromarray(img)  # Convert NumPy array to PIL Image
		if self.transform:
			img = self.transform(img)
        
        # Convert complementary labels to binary tensor here
		cl_binary = torch.zeros(self.num_classes, dtype=torch.float32)
		if isinstance(cl_label, (list, tuple, np.ndarray)):
			cl_binary[cl_label] = 1
		else:
			cl_binary[cl_label] = 1
            
		return img, cl_binary

# Split the dataset into training and validation sets
def split_dataset(data, train_ratio=0.9):
	train_data, val_data = {}, {}
	train_indices, val_indices = train_test_split(
		range(len(data['images'])), test_size=1-train_ratio, random_state=42
	)

	train_data['images'] = [data['images'][i] for i in train_indices]
	train_data['cl_labels'] = [data['cl_labels'][i] for i in train_indices]

	val_data['images'] = [data['images'][i] for i in val_indices]
	val_data['cl_labels'] = [data['cl_labels'][i] for i in val_indices]

	return train_data, val_data

# Split the dataset
# data_train, data_val = split_dataset(data)

# Define the MCL with EXP loss function
def mcl_exp_loss(output, cl_labels):
	exp_output = torch.exp(output)
	batch_size, num_classes = output.shape

	# Create a mask for the complementary labels
	mask = torch.zeros((batch_size, num_classes), device=output.device)
	for i, labels in enumerate(cl_labels):
		mask[i, labels] = 1

	# Compute the loss using the mask
	loss = -torch.mean(torch.log(1 - exp_output[mask.bool()]))
	return loss

def training_step(output, cl_labels):
        p = F.softmax(output, dim=1)
        p = ((1 - cl_labels) * p).sum(dim=1)
        loss = torch.exp(-p)
        loss = ((2 * 10 - 2) * loss / cl_labels.sum(dim=1)).sum()
        print("Train_Loss", loss)
        return loss

# Define the FWD loss function
def fwd_loss(output, cl_labels):
	softmax_output = torch.softmax(output, dim=1)
	loss = -torch.mean(torch.log(1 - softmax_output[range(len(cl_labels)), cl_labels]))
	return loss

# Evaluate the model
def evaluate_model(model, testloader, device):
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for images, labels in testloader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	accuracy = 100 * correct / total
	print(f'Test Accuracy: {accuracy:.2f}%')

def main():
	os.makedirs('./data/clcifar10', exist_ok=True)
	# Download the dataset if it doesn't exist
	dataset_path='./data/clcifar10/clcifar10.pkl'
	if not os.path.exists(dataset_path):
		gdown.download(id="1uNLqmRUkHzZGiSsCtV2-fHoDbtKPnVt2", output=dataset_path)

	# Read the dataset
	print("Loading dataset...")
	data = pickle.load(open(dataset_path, 'rb'))

	# Set up train and test transformations
	train_transform = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.RandomCrop(32, padding=4),
		transforms.ToTensor(),
		transforms.Normalize(
			[0.5068, 0.4854, 0.4402], [0.2672, 0.2563, 0.2760]
		),
	])
	test_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(
			[0.4922, 0.4832, 0.4486], [0.2456, 0.2419, 0.2605]
		),
	])

	# Prepare the dataset and dataloader
	trainset = CLCIFAR10(data, transform=train_transform)
	trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
	testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

	# Initialize the model
	model = models.resnet18(weights=None)
	model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust for 10 classes

	# Set up optimizer and loss function
	optimizer = optim.Adam(model.parameters(), lr=1e-4)
	loss_function = training_step  # Change to fwd_loss if needed

	# Select device
	if torch.cuda.is_available():
		# Use CUDA if available
		print("Using CUDA backend")
		device = torch.device("cuda")
	elif torch.backends.mps.is_available():
		# Use MPS if available
		print("Using MPS backend")
		device = torch.device("mps")
	else:
		# Fallback to CPU
		print("Using CPU backend")
		device = torch.device("cpu")
	model.to(device)

	# Training loop
	num_epochs = 10
	for epoch in range(num_epochs):
		print(f"Epoch {epoch+1}/{num_epochs}")
		model.train()
		running_loss = 0.0
		num_images = 0
		for images, cl_labels in trainloader:
			print(f"Images shape: {images.shape}")
			print(f"CL Labels shape: {cl_labels.shape}")  # Should now be [batch_size, num_classes]
            
			images = images.to(device)
			cl_labels = cl_labels.to(device)

            # Forward pass
			outputs = model(images)
			loss = loss_function(outputs, cl_labels)
            
            # Backward pass and optimization
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			print(f"Batch Loss: {loss.item():.4f}")
			running_loss += loss.item() * images.size(0)
			print(f"Batch Size: {images.size(0)}")
			print(f"Running Loss: {running_loss:.4f}")
			num_images += images.size(0)
			print(f"Number of Images Processed: {num_images}")

		print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/num_images:.4f}")

	# Call the evaluation function after training
	evaluate_model(model, testloader, device)

if __name__ == '__main__':
    main()