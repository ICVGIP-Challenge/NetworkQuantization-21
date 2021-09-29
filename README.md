# Post-training Data-Free Quantization Challenge-2021
![Block](misc/NetQ.png)
### Post-training Quantization
Convolutional Neural Networks (CNNs) have achieved tremendous results over prior-based arts. 
However, deploying them on real-time resource-constrained devices that operate on the restricted latency and computational budget is not always practically feasible. Generally, number of parameters and size of the model inhibit efficient inference on such devices. One way to make such models more resource efficient is by performing quantization on pretrained models. 


To aid with the above, this repository contains baseline codes for quantization of weights and activations of deep learning-based models.


Quantization converts the floating-point (FP32) weights and activations of a deep CNN to low-bit integers (*e.g.,* INT8). 
The conversion not only reduces the memory footprint but also improves the computational latency of the model.
This challenge focuses on *Post-training quantization*, where a pretrained FP32 model is provided as input for quantization, and no re-training is required.


Quantization requires two parameters to be computed for an input tensor, (a) scale and (b) zero-point offset.
The computation of scale and zero-point offset relies on the min-max range of the input tensor.
The range of the FP32 weights can be directly computed. Whereas, for activations, original training samples or their subset may be required.
If the original training samples (in full or subset) are available, they can be used to generate the activations and obtain the range. 
Once the range calibration of the activations is complete, corresponding scale and zero-point offset can be computed as well. 
The recorded scale and zero-point offset of the weights and activations can be then used to quantize the entire network before inference (static mode).


For details on quantization and various related techniques, readers may refer to [Survey Paper 1](https://arxiv.org/abs/1806.08342), [Survey Paper 2](https://arxiv.org/abs/2103.13630). 

### Problem Statement
The availability of the original training dataset in full or subset may not be feasible for some tasks, such as medical imaging, where the user's privacy is prioritized above all. Therefore the challenge is to perform range calibration of the activations without using original training samples. Precisely, **the task is to perform post-training quantization of the pretrained FP32 models for imagenet classification without using any data (data-free)**.

Some references on post-training data-free quantization are [DFQ](https://arxiv.org/abs/1906.04721), [ZeroQ](https://arxiv.org/abs/2001.00281), [GDFQ](https://arxiv.org/abs/2003.03603).

### Code
This repository contains a baseline to get started on the task. 

System requirements
```bash
Pytorch 1.8.0
Torchvision 0.9.0
Pillow 
progress.bar
*No GPU support is required.*
```
- ```main.py``` provides the baseline codes with detailed instructions for post-training static data-free quantization in Pytorch framework. 
- Participants are suggested to go through the files and comments for a better understanding of the framework.
- For simplicity, we have restricted the quantization configuration to 8-bit integers for FP32 weights and activations. Participants can modify the baseline codes to change accordingly to quantize a layer or a subset of them.
- This will help in maintaining the trade-off between size and accuracy of quantized model against FP32 baseline.

Note that we have used randomly generated samples from unit Gaussian in the baseline code (instead of using original training samples for range calibration of the activations). The participants are required to propose a new representative/artificial dataset that can be used for range calibration of the activations instead of unit Gaussian and original training samples. For this, only a specified section of the code is needed to be modified. The corresponding sections and details are provided in the baseline script `main.py`. 

### Dataset
The challenge requires Imagenet validation set (ILSVRC 2012). It can be downloaded from [link](https://image-net.org/challenges/LSVRC/2012/).

### Usage
- The baseline code supports the INT8 quantization of the following pretrained models on Imagenet:

    1. ResNet18
    2. MobilenetV2
    3. ShuffleNetV2

- *Do not change the quantization scheme in the baseline code. It may affect the final ranking*.
- Set the imagenet validation set path in `Line 21` of the file `dataset.py`.
- Run the following command to get initial results of the above models after quantization without using original training samples:
```bash
bash run.bash > ./output.txt
``` 

Above command will generate the ```output.txt``` file which will have the predicted class label for each image in the validation set of the Imagenet dataset.

#### Demo Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16cYmpPUCc5lrf3Z9gx1W_WM2ThcoSZki?usp=sharing)
We also include a Colab notebook with a demo showing how to run the baseline code to quantize a FP32 model and test on a single image.

### Submission
#### Validation Phase
- Each participating team is required to upload only the generated `output.txt` file during the validation phase.
- *Note: Do not change the output format of the baseline code. The output.txt file will be parsed to compute the ranking during validation phase.*

#### Testing Phase
- During test phase, participants need to upload the following in .zip format:

    1. `output.txt`
    2. Code (*with clear comments where the modification has been done*)
    3. quantized models for the supported/allowed models in the baseline code.

- We will announce further instructions for testing phase in due course of time.

#### Evaluation Metric
The submissions will be judged for maximum compression and minimal drop in performance. The methods must have a compression ratio of more than 25%.
```
Compression ratio = 100 * (orignial_model_size - new_model_size) / orignial_model_size
```
The entries with compression ratio more than 25% will be sorted by the compression ratio first and then by accuracy, i.e. method with higher compression ratio for a given accuracy will win. The precision for accuracy will be 0.1%, i.e. accuracies of 80.13% and 80.14% will be considered the same, and that of 88.15% (which rounds to 88.2%) will be considered higher.

### Ranking
- The `output.txt` file will be used to compute the ranking during validation phase.

More details on testing and ranking methodology will be made available during the respective phases of competition. 

### Contact us
- Please use the discussion/issues section of this repository to post your queries. 
