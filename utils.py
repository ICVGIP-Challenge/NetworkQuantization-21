# credits:
# ------------
# partly adapted from
# https://github.com/amirgholami/ZeroQ/blob/master/classification/utils/train_utils.py

# regular imports 
import os
from PIL import Image
import sys
import time

# torch imports 
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import torchvision.models.quantization as models

# relative imports 
from dataset import *
from progress.bar import Bar

# DO NOT CHANGE THIS
# urls of pretrained imagenet fp32 models
model_urls = {
				'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
				'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
				'shufflenet_v2_x1_0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth'
			}

# for loading the pretrained imagenet fp32 weights 
def get_model_from_zoo(model_name):
	# quantize=False means <don't donwload the already quantized version of the model from the zoo>
	if model_name == 'resnet18':
		model = models.resnet18(pretrained=False, quantize=False)
	elif model_name == 'mobilenet_v2':
		model = models.mobilenet_v2(pretrained=False, quantize=False)
	elif model_name == 'shufflenet_v2_x1_0':
		model = models.shufflenet_v2_x1_0(pretrained=False, quantize=False)
	
	checkpoint_url = model_urls[model_name]
	model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_url, progress=True))
	return model 

# testing code
# test a quantized model on given dataset
# for imagenet classification
# DO NOT MAKE CHANGES IN THIS CODE 
def test(model, test_loader, device):
	"""
	test a model on a given dataset
	"""
	model.eval()
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(test_loader):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = model(inputs)
			_, predicted = outputs.max(1)
			for p in list(predicted.numpy()):
				print(p)
	return None
			
# for checking the size of the original/quantized model
def print_size_of_model(model):
	torch.save(model.state_dict(), "temp.p")
	print('In (MB):', os.path.getsize("temp.p")/1e6)
	os.remove('temp.p')

# for range calibration of the activation during quantization
def evaluate(model, data_loader, neval_batches):
	model.eval()
	cnt=0
	with torch.no_grad():
		for image in data_loader:
			output = model(image)
			cnt+=1
			if cnt >= neval_batches:
				 return None
	return None