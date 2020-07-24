# ----------------------------------------------------- #
# Example of how to use object-centric ISR RGB-D Dataset
# Task: Object classification
# Framework: PyTorch
# CNN: AlexNet
# ----------------------------------------------------- #
import numpy as np
import cv2
import os
import argparse
from isr_rgbd_dataset import *
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
import torchsummary


class Dataset(data.Dataset):
	'Characterizes a dataset for PyTorch'
	def __init__(self, list_IDs, labels):
		'Initialization'
		self.labels = labels
		self.list_IDs = list_IDs

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.list_IDs)

	def __getitem__(self, index):
		'Generates one sample of data'
		# Select sample
		ID = self.list_IDs[index,:,:,:]

		#Load data and get label
		X = torch.from_numpy(ID).float()
		X_data = X.permute(2,0,1)

		y = (np.long(self.labels[index]))

		return X_data, y


def cnn_training(config, network, trainloader):
	training_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	cnn_model = network.to(training_device)
	print("\nNetwork details: ")
	torchsummary.summary(cnn_model, (3, 224, 224))

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(cnn_model.parameters(), lr = config.lr, weight_decay = 0.0005)

	# Decreasing learning rate
	learning_rate_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config.epochs)

	# Network's training
	print("\nNetwork training...\n")
	for epoch in range(config.epochs):
		learning_rate_scheduler.step()
		print(' ')

		for local_batch, data in enumerate(trainloader, 0):
			images, labels = data
			images, labels = images.to(training_device), labels.to(training_device)

			optimizer.zero_grad()
			predictions = cnn_model(images)
			loss = criterion(predictions, labels)
			loss.backward()
			optimizer.step()

			print ('[%d, %5d] loss: %.3f with lr = %.7f' % (epoch + 1, local_batch + 1,  loss.item(), learning_rate_scheduler.get_lr()[0]))

	return cnn_model


def cnn_testing(network, testloader, object_classes):
	testing_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	cnn_model = network.to(testing_device)
	cnn_model.eval()

	correct = 0
	total = 0
	class_correct = list(0. for i in range(len(object_classes)))
	class_total = list(0. for i in range(len(object_classes)))
	conf_matrix =[ [0 for x in range( len(object_classes) )] for y in range( len(object_classes) ) ]

	with torch.no_grad():
		for local_batch, data in enumerate(testloader,0):
			images, labels = data
			images, labels = images.to(testing_device), labels.to(testing_device)

			predictions = cnn_model(images)
			prob, predicted = torch.max(predictions.data, 1)

			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			c = (predicted == labels).squeeze()

			for i in range(len(images)):
				label = labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1


			conf_matrix+=confusion_matrix(predicted.cpu(), labels.cpu(),labels=[x for x in range( len(object_classes) )])

	print('\nTest Accuracy of the model on the {} test images: {} %\n'.format(total,100 * correct / total))
	print(conf_matrix)

	for i in range(len(object_classes)):
		print('Accuracy of %5s : %2d %% in %d Images' % (object_classes[i], 100 * class_correct[i] / class_total[i], class_total[i]))



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", type=int, default=25, help="number of training epochs")
	parser.add_argument("--batch_size", type=int, default=32, help="batch size")
	parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
	parser.add_argument("--isr_class_names_path", type=str, default=os.path.join('isr_rgbd_dataset.names'))
	parser.add_argument("--isr_rgb_images_path", type=str, default=os.path.join('RGB',''))
	parser.add_argument("--raw_images_shuffle", default=True)
	parser.add_argument("--object_centric_image_size", type=int, default=224)
	opt = parser.parse_args()

	# -------------------------------------- Dataset -------------------------------------- #
	# Get ISR Object-centric RGB-D Images
	print("\nGetting ISR Object-centric RGB-D Images")
	object_classes_names = open(opt.isr_class_names_path).read().split("\n")[:-1]
	print("Number of classes: " + str(len(object_classes_names)))
	rgb_training_raw_imgs_path, rgb_testing_raw_imgs_path =  raw_images_training_test_split(opt)
	print("\nTraining data...")
	rgb_training_object_imgs, depth_training_object_imgs, training_object_labels = get_object_centric_images(opt, rgb_training_raw_imgs_path)
	print("\nTesting data...")
	rgb_testing_object_imgs, depth_testing_object_imgs, testing_object_labels = get_object_centric_images(opt, rgb_testing_raw_imgs_path)

	# Prepare the dataset for PyTorch framework
	training_set = Dataset(rgb_training_object_imgs, training_object_labels)
	test_set = Dataset(rgb_testing_object_imgs, testing_object_labels)

	train_loader = data.DataLoader(dataset=training_set, batch_size = opt.batch_size, shuffle = True)
	test_loader = data.DataLoader(dataset=test_set, batch_size = opt.batch_size, shuffle = False)
	# ------------------------------------ End Dataset ------------------------------------ #

	# -------------------------------------- Network -------------------------------------- #
	CNN_AlexNet = models.alexnet(pretrained = True)
	CNN_AlexNet.classifier[6] = nn.Linear(in_features = 4096, out_features = len(object_classes_names))

	CNN_AlexNet = cnn_training(opt, CNN_AlexNet, train_loader)
	cnn_testing(CNN_AlexNet, test_loader, object_classes_names)
	# ------------------------------------ End Network ------------------------------------ #



