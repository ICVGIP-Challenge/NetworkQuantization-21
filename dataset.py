# regular imports 
import numpy as np

# torch imports 
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

generator = torch.Generator()

# dataloaders for imagenet dataset
def get_testloader(dataset='imagenet', bs=None):    
    if dataset == 'imagenet':
        # except inception networks
        input_size = 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # set the imagenet validation path 
        data_path = 'path to imagenet validation dataset ILSVRC 2012'
        test_dataset = datasets.ImageFolder(
                                            data_path,
                                            transforms.Compose([
                                                transforms.Resize(int(input_size / 0.875)),
                                                transforms.CenterCrop(input_size),
                                                transforms.ToTensor(),
                                                normalize,
                                          ]))

        test_sampler = torch.utils.data.RandomSampler(test_dataset, generator=generator.manual_seed(1562))
        test_loader = DataLoader(test_dataset,
                                 batch_size=bs,
                                 shuffle=False,
                                 sampler=test_sampler,
                                 num_workers=32)
    return test_loader


# using unit Gaussian samples
# for range calibration of the activations
# on the models for ImageNet classification
class UniformDataset(Dataset):
    """
    get Gaussian samples with mean 0 and variance 1
    """
    def __init__(self, length, size, transform):
        self.length = length
        self.transform = transform
        self.size = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        #generating gaussian data of shape 'size'      
        sample = torch.randn(size = self.size)
        return sample
'''
returns the dataloader for Gaussian samples to calibrate 
the range of the activations for quantization.
'''
def getGaussianData(dataset='cifar10', batch_size=32):
    if dataset == 'imagenet':
        # except inception networks
        size = (3, 224, 224)
        num_data = 1000
    dataset = UniformDataset(length=num_data, size=size, transform=None) 
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=32)
    return data_loader