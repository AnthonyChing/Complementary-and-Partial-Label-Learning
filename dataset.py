import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision
from sklearn.model_selection import train_test_split
import random

class CLCIFAR10(Dataset):
	"""CLCIFAR10 training set
	The training set of CIFAR10 with human annotated complementary labels.
	Containing 50000 samples, each with one ordinary label and the first one of the three complementary labels
	"""
	def __init__(self, data, transform=None, num_classes=10, train=True, invert_labels=False, algo=None):
		self.transform = transform
		self.num_classes = num_classes
		self.invert_labels = invert_labels
		self.algo = algo
		# Preprocess images and labels
		self.images = [self.transform(Image.fromarray(img)) if self.transform else Image.fromarray(img) for img in data['images']]
		self.labels = []
		if train:
			for cl_label in data['cl_labels']:
				cl_binary = torch.zeros(self.num_classes, dtype=torch.float32)
				if isinstance(cl_label, (list, tuple, np.ndarray)):
					cl_binary[cl_label] = 1
				else:
					cl_binary[cl_label] = 1
				if self.invert_labels:
					cl_binary = 1 - cl_binary
				self.labels.append(cl_binary)
		else:
			for ord_label in data['ord_labels']:
				cl_binary = torch.zeros(self.num_classes, dtype=torch.float32)
				if isinstance(ord_label, (list, tuple, np.ndarray)):
					cl_binary[ord_label] = 1
				else:
					cl_binary[ord_label] = 1
				if self.invert_labels:
					cl_binary = 1 - cl_binary
				self.labels.append(cl_binary)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		if self.algo == 'proden':
			return self.images[idx], self.labels[idx], idx
		else:
			return self.images[idx], self.labels[idx]

def _cifar100_to_cifar20(target):
	_dict = {
		0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3,
		10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11,
		20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15,
		30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5,
		40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10,
		50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17,
		60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19,
		70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13,
		80: 16, 81: 19, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19,
		90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13,
	}
	return _dict[target]

class CLCIFAR20(torchvision.datasets.CIFAR100, Dataset):
	"""CLCIFAR20 training set
	The training set of CIFAR20 with human annotated complementary labels.
	Containing 50000 samples, each with one ordinary label and the first one of the three complementary labels
	"""
	def __init__(self, data=None, transform=None, num_classes=20, train=True, invert_labels=False, algo=None):
		self.transform = transform
		self.num_classes = num_classes
		self.train = train
		self.invert_labels = invert_labels
		self.algo = algo
		if not train:
			super(CLCIFAR20, self).__init__(
				root='./data/cifar20', train=train, transform=transform, download=True
			)
		else:
			# Preprocess images and labels (no _cifar100_to_cifar20 conversion)
			self.images = [self.transform(Image.fromarray(img)) if self.transform else Image.fromarray(img) for img in data['images']]
			self.labels = []
			for cl_label in data['cl_labels']:
				cl_binary = torch.zeros(self.num_classes, dtype=torch.float32)
				if isinstance(cl_label, (list, tuple, np.ndarray)):
					cl_binary[cl_label] = 1
				else:
					cl_binary[cl_label] = 1
				if self.invert_labels:
					cl_binary = 1 - cl_binary
				self.labels.append(cl_binary)

	def __len__(self):
		if self.train:
			return len(self.images)
		else:
			return super().__len__()

	def __getitem__(self, idx):
		if self.train:
			if self.algo == 'proden':
				return self.images[idx], self.labels[idx], idx
			else:
				return self.images[idx], self.labels[idx]
		else:
			img, target = super().__getitem__(idx)
			# Convert CIFAR100 label to CIFAR20 label for test set
			target_20 = _cifar100_to_cifar20(target)
			cl_binary = torch.zeros(self.num_classes, dtype=torch.float32)
			cl_binary[target_20] = 1
			if self.invert_labels:
				cl_binary = 1 - cl_binary
			return img, cl_binary

class CLMIN10(Dataset):
	def __init__(self, data=None, transform=None, num_classes=10, train=True, invert_labels=False, algo=None):
		self.transform = transform
		self.num_classes = num_classes
		self.train = train
		self.invert_labels = invert_labels
		self.algo = algo
		self.images = [self.transform(Image.fromarray(img)) if self.transform else Image.fromarray(img) for img in data['images']]
		self.labels = []
		if train:
			for cl_label in data['cl_labels']:
				cl_binary = torch.zeros(self.num_classes, dtype=torch.float32)
				if isinstance(cl_label, (list, tuple, np.ndarray)):
					cl_binary[cl_label] = 1
				else:
					cl_binary[cl_label] = 1
				if self.invert_labels:
					cl_binary = 1 - cl_binary
				self.labels.append(cl_binary)
		else:
			for ord_label in data['ord_labels']:
				cl_binary = torch.zeros(self.num_classes, dtype=torch.float32)
				if isinstance(ord_label, (list, tuple, np.ndarray)):
					cl_binary[ord_label] = 1
				else:
					cl_binary[ord_label] = 1
				if self.invert_labels:
					cl_binary = 1 - cl_binary
				self.labels.append(cl_binary)
	def __len__(self):
		return len(self.images)
	def __getitem__(self, idx):
		if self.algo == 'proden':
			return self.images[idx], self.labels[idx], idx
		else:
			return self.images[idx], self.labels[idx]

class CLMIN20(Dataset):
	def __init__(self, data=None, transform=None, num_classes=20, train=True, invert_labels=False, algo=None):
		self.transform = transform
		self.num_classes = num_classes
		self.train = train
		self.invert_labels = invert_labels
		self.algo = algo
		self.images = [self.transform(Image.fromarray(img)) if self.transform else Image.fromarray(img) for img in data['images']]
		self.labels = []
		if train:
			for cl_label in data['cl_labels']:
				cl_binary = torch.zeros(self.num_classes, dtype=torch.float32)
				if isinstance(cl_label, (list, tuple, np.ndarray)):
					cl_binary[cl_label] = 1
				else:
					cl_binary[cl_label] = 1
				if self.invert_labels:
					cl_binary = 1 - cl_binary
				self.labels.append(cl_binary)
		else:
			for ord_label in data['ord_labels']:
				cl_binary = torch.zeros(self.num_classes, dtype=torch.float32)
				if isinstance(ord_label, (list, tuple, np.ndarray)):
					cl_binary[ord_label] = 1
				else:
					cl_binary[ord_label] = 1
				if self.invert_labels:
					cl_binary = 1 - cl_binary
				self.labels.append(cl_binary)
	def __len__(self):
		return len(self.images)
	def __getitem__(self, idx):
		if self.algo == 'proden':
			return self.images[idx], self.labels[idx], idx
		else:
			return self.images[idx], self.labels[idx]

class PLCIFAR10(Dataset):
    """Partial Label CIFAR10 dataset with uniform candidate label noise."""
    def __init__(self, data, transform=None, num_classes=10, train=True, invert_labels=False, algo=None, q=0.1):
        self.transform = transform
        self.num_classes = num_classes
        self.invert_labels = invert_labels
        self.algo = algo
        self.q = q
        self.images = [self.transform(Image.fromarray(img)) if self.transform else Image.fromarray(img) for img in data['images']]
        self.labels = []
        if train:
            for ord_label in data['ord_labels']:
                candidate = set([ord_label])
                for c in range(self.num_classes):
                    if c != ord_label and random.random() < self.q:
                        candidate.add(c)
                # Ensure at least two labels
                if len(candidate) == 1:
                    incorrect = random.choice([c for c in range(self.num_classes) if c != ord_label])
                    candidate.add(incorrect)
                cl_binary = torch.zeros(self.num_classes, dtype=torch.float32)
                cl_binary[list(candidate)] = 1
                if self.invert_labels:
                    cl_binary = 1 - cl_binary
                self.labels.append(cl_binary)
        else:
            for ord_label in data['ord_labels']:
                cl_binary = torch.zeros(self.num_classes, dtype=torch.float32)
                cl_binary[ord_label] = 1
                if self.invert_labels:
                    cl_binary = 1 - cl_binary
                self.labels.append(cl_binary)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        if self.algo == 'proden':
            return self.images[idx], self.labels[idx], idx
        else:
            return self.images[idx], self.labels[idx]

class PLCIFAR20(torchvision.datasets.CIFAR100, Dataset):
    """Partial Label CIFAR20 dataset with uniform candidate label noise."""
    def __init__(self, data=None, transform=None, num_classes=20, train=True, invert_labels=False, algo=None, q=0.1):
        self.transform = transform
        self.num_classes = num_classes
        self.train = train
        self.invert_labels = invert_labels
        self.algo = algo
        self.q = q
        if not train:
            super(PLCIFAR20, self).__init__(
                root='./data/cifar20', train=train, transform=transform, download=True
            )
        else:
            self.images = [self.transform(Image.fromarray(img)) if self.transform else Image.fromarray(img) for img in data['images']]
            self.labels = []
            for ord_label in data['ord_labels']:
                candidate = set([ord_label])
                for c in range(self.num_classes):
                    if c != ord_label and random.random() < self.q:
                        candidate.add(c)
                if len(candidate) == 1:
                    incorrect = random.choice([c for c in range(self.num_classes) if c != ord_label])
                    candidate.add(incorrect)
                cl_binary = torch.zeros(self.num_classes, dtype=torch.float32)
                cl_binary[list(candidate)] = 1
                if self.invert_labels:
                    cl_binary = 1 - cl_binary
                self.labels.append(cl_binary)
    def __len__(self):
        if self.train:
            return len(self.images)
        else:
            return super().__len__()
    def __getitem__(self, idx):
        if self.train:
            if self.algo == 'proden':
                return self.images[idx], self.labels[idx], idx
            else:
                return self.images[idx], self.labels[idx]
        else:
            img, target = super().__getitem__(idx)
            target_20 = _cifar100_to_cifar20(target)
            cl_binary = torch.zeros(self.num_classes, dtype=torch.float32)
            cl_binary[target_20] = 1
            if self.invert_labels:
                cl_binary = 1 - cl_binary
            return img, cl_binary

class PLMIN10(Dataset):
    def __init__(self, data=None, transform=None, num_classes=10, train=True, invert_labels=False, algo=None, q=0.1):
        self.transform = transform
        self.num_classes = num_classes
        self.train = train
        self.invert_labels = invert_labels
        self.algo = algo
        self.q = q
        self.images = [self.transform(Image.fromarray(img)) if self.transform else Image.fromarray(img) for img in data['images']]
        self.labels = []
        if train:
            for ord_label in data['ord_labels']:
                candidate = set([ord_label])
                for c in range(self.num_classes):
                    if c != ord_label and random.random() < self.q:
                        candidate.add(c)
                if len(candidate) == 1:
                    incorrect = random.choice([c for c in range(self.num_classes) if c != ord_label])
                    candidate.add(incorrect)
                cl_binary = torch.zeros(self.num_classes, dtype=torch.float32)
                cl_binary[list(candidate)] = 1
                if self.invert_labels:
                    cl_binary = 1 - cl_binary
                self.labels.append(cl_binary)
        else:
            for ord_label in data['ord_labels']:
                cl_binary = torch.zeros(self.num_classes, dtype=torch.float32)
                cl_binary[ord_label] = 1
                if self.invert_labels:
                    cl_binary = 1 - cl_binary
                self.labels.append(cl_binary)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        if self.algo == 'proden':
            return self.images[idx], self.labels[idx], idx
        else:
            return self.images[idx], self.labels[idx]

class PLMIN20(Dataset):
    def __init__(self, data=None, transform=None, num_classes=20, train=True, invert_labels=False, algo=None, q=0.1):
        self.transform = transform
        self.num_classes = num_classes
        self.train = train
        self.invert_labels = invert_labels
        self.algo = algo
        self.q = q
        self.images = [self.transform(Image.fromarray(img)) if self.transform else Image.fromarray(img) for img in data['images']]
        self.labels = []
        if train:
            for ord_label in data['ord_labels']:
                candidate = set([ord_label])
                for c in range(self.num_classes):
                    if c != ord_label and random.random() < self.q:
                        candidate.add(c)
                if len(candidate) == 1:
                    incorrect = random.choice([c for c in range(self.num_classes) if c != ord_label])
                    candidate.add(incorrect)
                cl_binary = torch.zeros(self.num_classes, dtype=torch.float32)
                cl_binary[list(candidate)] = 1
                if self.invert_labels:
                    cl_binary = 1 - cl_binary
                self.labels.append(cl_binary)
        else:
            for ord_label in data['ord_labels']:
                cl_binary = torch.zeros(self.num_classes, dtype=torch.float32)
                cl_binary[ord_label] = 1
                if self.invert_labels:
                    cl_binary = 1 - cl_binary
                self.labels.append(cl_binary)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        if self.algo == 'proden':
            return self.images[idx], self.labels[idx], idx
        else:
            return self.images[idx], self.labels[idx]

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

def get_dataset(args, invert_labels=False):
	import os
	import pickle
	import gdown
	from torchvision import transforms

	if args.ds == 'clcifar20':
		os.makedirs('./data/clcifar20', exist_ok=True)
		dataset_path = './data/clcifar20/clcifar20.pkl'
		if not os.path.exists(dataset_path):
			gdown.download(id="1PhZsyoi1dAHDGlmB4QIJvDHLf_JBsFeP", output=dataset_path)
		num_classes = 20
		# print("Loading dataset...")
		data = pickle.load(open(dataset_path, 'rb'))
		train_transform = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(32, padding=4),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
			),
		])
		test_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(
				[0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
			),
		])
		trainset = CLCIFAR20(data, transform=train_transform, invert_labels=invert_labels, algo=args.algo)
		testset = CLCIFAR20(transform=test_transform, train=False)
		return trainset, testset, num_classes
	elif args.ds == 'clcifar10':
		os.makedirs('./data/clcifar10', exist_ok=True)
		dataset_path = './data/clcifar10/clcifar10.pkl'
		if not os.path.exists(dataset_path):
			gdown.download(id="1uNLqmRUkHzZGiSsCtV2-fHoDbtKPnVt2", output=dataset_path)
		num_classes = 10
		# print("Loading dataset...")
		data = pickle.load(open(dataset_path, 'rb'))
		train_transform = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(32, padding=4),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
			),
		])
		test_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(
				[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
			),
		])
		trainset = CLCIFAR10(data, transform=train_transform, invert_labels=invert_labels, algo=args.algo)
		testset = CLCIFAR10(data, transform=test_transform, train=False)
		return trainset, testset, num_classes
	elif args.ds == 'clmin10':
		os.makedirs('./data/clmin10', exist_ok=True)
		# Build train set
		dataset_path = './data/clmin10/clmin10_train.pkl'
		if not os.path.exists(dataset_path):
			gdown.download(id="1k02mwMpnBUM9de7TiJLBaCuS8myGuYFx", output=dataset_path)
		num_classes = 10
		# print("Loading dataset...")
		data = pickle.load(open(dataset_path, 'rb'))
		train_transform = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(64, padding=8),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
			),
		])
		trainset = CLMIN10(data, transform=train_transform, invert_labels=invert_labels, algo=args.algo)
		# Build test set
		dataset_path = './data/clmin10/clmin10_test.pkl'
		if not os.path.exists(dataset_path):
			gdown.download(id="1e8fZN8swbg9wc6BSOC0A5KHIqCY2C7me", output=dataset_path)
		test_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
			),
		])
		testset = CLMIN10(data, transform=test_transform, train=False)
		return trainset, testset, num_classes
	elif args.ds == 'clmin20':
		os.makedirs('./data/clmin20', exist_ok=True)
		# Build train set
		dataset_path = './data/clmin20/clmin20_train.pkl'
		if not os.path.exists(dataset_path):
			gdown.download(id="1Urdxs_QTxbb1gDBpmjP09Q35btckI3_d", output=dataset_path)
		num_classes = 20
		# print("Loading dataset...")
		data = pickle.load(open(dataset_path, 'rb'))
		train_transform = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(64, padding=8),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
			),
		])
		trainset = CLMIN20(data, transform=train_transform, invert_labels=invert_labels, algo=args.algo)
		# Build test set
		dataset_path = './data/clmin20/clmin20_test.pkl'
		if not os.path.exists(dataset_path):
			gdown.download(id="1EdBCrifSrIIUg1ioPWA-ZLEHO53P4NPl", output=dataset_path)
		test_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
			),
		])
		testset = CLMIN20(data, transform=test_transform, train=False)
		return trainset, testset, num_classes
	elif args.ds == 'cifar10_pl':
		os.makedirs('./data/clcifar10', exist_ok=True)
		dataset_path = './data/clcifar10/clcifar10.pkl'
		if not os.path.exists(dataset_path):
			gdown.download(id="1uNLqmRUkHzZGiSsCtV2-fHoDbtKPnVt2", output=dataset_path)
		num_classes = 10
		data = pickle.load(open(dataset_path, 'rb'))
		train_transform = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(32, padding=4),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
			),
		])
		test_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(
				[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
			),
		])
		trainset = PLCIFAR10(data, transform=train_transform, invert_labels=invert_labels, algo=args.algo)
		testset = PLCIFAR10(data, transform=test_transform, train=False)
		return trainset, testset, num_classes
	elif args.ds == 'cifar20_pl':
		os.makedirs('./data/clcifar20', exist_ok=True)
		dataset_path = './data/clcifar20/clcifar20.pkl'
		if not os.path.exists(dataset_path):
			gdown.download(id="1PhZsyoi1dAHDGlmB4QIJvDHLf_JBsFeP", output=dataset_path)
		num_classes = 20
		data = pickle.load(open(dataset_path, 'rb'))
		train_transform = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(32, padding=4),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
			),
		])
		test_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(
				[0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
			),
		])
		trainset = PLCIFAR20(data, transform=train_transform, invert_labels=invert_labels, algo=args.algo)
		testset = PLCIFAR20(transform=test_transform, train=False)
		return trainset, testset, num_classes
	elif args.ds == 'min10_pl':
		os.makedirs('./data/clmin10', exist_ok=True)
		dataset_path = './data/clmin10/clmin10_train.pkl'
		if not os.path.exists(dataset_path):
			gdown.download(id="1k02mwMpnBUM9de7TiJLBaCuS8myGuYFx", output=dataset_path)
		num_classes = 10
		data = pickle.load(open(dataset_path, 'rb'))
		train_transform = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(64, padding=8),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
			),
		])
		trainset = PLMIN10(data, transform=train_transform, invert_labels=invert_labels, algo=args.algo)
		dataset_path = './data/clmin10/clmin10_test.pkl'
		if not os.path.exists(dataset_path):
			gdown.download(id="1e8fZN8swbg9wc6BSOC0A5KHIqCY2C7me", output=dataset_path)
		test_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
			),
		])
		testset = PLMIN10(data, transform=test_transform, train=False)
		return trainset, testset, num_classes
	elif args.ds == 'min20_pl':
		os.makedirs('./data/clmin20', exist_ok=True)
		dataset_path = './data/clmin20/clmin20_train.pkl'
		if not os.path.exists(dataset_path):
			gdown.download(id="1Urdxs_QTxbb1gDBpmjP09Q35btckI3_d", output=dataset_path)
		num_classes = 20
		data = pickle.load(open(dataset_path, 'rb'))
		train_transform = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(64, padding=8),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
			),
		])
		trainset = PLMIN20(data, transform=train_transform, invert_labels=invert_labels, algo=args.algo)
		dataset_path = './data/clmin20/clmin20_test.pkl'
		if not os.path.exists(dataset_path):
			gdown.download(id="1EdBCrifSrIIUg1ioPWA-ZLEHO53P4NPl", output=dataset_path)
		test_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
			),
		])
		testset = PLMIN20(data, transform=test_transform, train=False)
		return trainset, testset, num_classes
	else:
		raise NotImplementedError(f"Dataset {args.ds} not implemented yet.")
