# Post-training Static Data-free Quantization 
# of Deep Neural Networks in Pytorch Framework
# --------------------------------------------------------
# Datasets support: ImageNet (ILSVRC 2012 Validation)
# usage:
# ---------------------------
# bash run.bash > ./output.txt
# ---------------------------
# META INFO:

'''
 1. The output of the above bash file run is saved in ./output.txt.
 2. Quantized model is saved as '<model_name>_quantized_model.pth' in the './ckpts/<dataset>_quantized_models/'.
 3. During validation phase, upload only the generated output.txt file. 
 4. For testing phase upload instructions, keep a watch on challenge website or this repository. 
 3. Quantized inference is supported in CPU mode only. No need to have a GPU support. 
 4. The code supports only INT8 quantization config. Do not make changes in the quantization configuration.
 5. For ImageNet, it automatically downloads the FP32 models from torchvision model zoo.  
    #  No need to explicitly provide a pretrained FP32 model. This may impact your participation. 
 6. Do not [add/remove/comment/uncomment] the print statements in all the code files provided in this repo. It may impact your evaluation. 
'''

# regular imports
import argparse
import copy
import os

# torch imports
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torch.quantization

# custom utils import
from dataset import *
from utils import *

# fixed. 
torch.manual_seed(8021992)

# program setting
def arg_parse():
    parser = argparse.ArgumentParser(
        description='Sample codes for the quantization of deep neural networks.')

    parser.add_argument('--model',
                        type=str,
                        default='resnet18',
                        choices=['resnet18', 'mobilenet_v2', 'shufflenet_v2_x1_0'],
                        help='name/id of imagenet fp32 pretrained models')

    parser.add_argument('--dataset',
                        type=str,
                        default='imagenet',
                        choices=['imagenet'],
                        help='type of dataset')
                        
    parser.add_argument('--test_bs',
                        type=int,
                        default=128,
                        help='batch size of test data')

    args = parser.parse_args()
    return args

# main program
if __name__ == '__main__':
    args = arg_parse()

    # setting the device to be used for computation
    # CPU is supported and sufficient to run. Don't change it to 'cuda'.
    device = 'cpu'

    # initializing test dataloader for testing the quantized models
    test_loader = get_testloader(args.dataset, bs=args.test_bs)
    
    # pretrained FP32 model for imagenet dataset 
    model = get_model_from_zoo(args.model)
    model.to(device)
    model.eval()

    # fuse the Batch-Norm parameters with conv layers for efficient inference
    model.fuse_model()

    # make a copy for quantization
    orig_model = copy.deepcopy(model)

    # quantization begins
    # put the model on CPU first and set the eval mode on
    orig_model = orig_model.cpu()
    orig_model.eval()

    '''
    By formulation, quantization requires range of the input 
    tensor [fp32 weights/activations]. For weights, range 
    can be directly computed.

    Whereas for activations, one need to have access to 
    original data or subset of it. Once the dataset is 
    available, activations can be generated. Then the 
    range can be recorded and step size and zero-point 
    can be calculated. Then these recorded stats are 
    used for quantization of activations during inference.  

    This challenge aims to come up with solutions that 
    do not require access to original dataset or its subset 
    in any form for the range calibration of the 
    activations. 

    Hint: One can utilize a representative dataset that will be used 
    to generate the activations. Once the activations are generated, 
    range can be recorded and activations can be quantized during 
    inference.

    Here you can add the codes for data-free quantization 
    to get the artificial data that will replace the 
    original dataset during range calibration in later
    part of this code ([Line: 138, main.py]).

    By default, this code will use some random samples from 
    unit Gaussian distribution for the same.
    '''

    # get gaussain dataloader for range calibration of the activations
    gaussian_dataloader = getGaussianData(args.dataset)

    # number of batches to be used for range calibration of the activations for quantization
    num_calibration_batches = 32

    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    orig_model.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(orig_model, inplace=True)

    # Range calibrate first
    # Calibrating the ranges of activations with the training set
    evaluate(orig_model, gaussian_dataloader, neval_batches=num_calibration_batches)

    # Convert to quantized model
    torch.quantization.convert(orig_model, inplace=True)

    print_size_of_model(orig_model)

    # print('Accuracy using quantized INT8 model')
    test(orig_model, test_loader, device)

    # saving the quantized weights for future inference
    save_dir = './ckpts/'+str(args.dataset)+'_quantized_models/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.jit.save(torch.jit.script(orig_model), save_dir+args.model+'_quantized_model.pth')
    # for loading the quantized model
    # torch.jit.load(quantized_model_file) or 
    # # use load_quantized_weights function from utils.py

    # # also save fp32 model in the same dir for future reference to compute speedup achieved
    # torch.jit.save(torch.jit.script(model), save_dir + args.model+'_fp32_model.pth')