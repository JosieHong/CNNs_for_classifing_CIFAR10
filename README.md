# CNNs_for_classifing_CIFAR10

 Implement of LeNet, AlexNet, VGG16, GoogLeNet, ResNet50 for classifing on CIFAR10 with tensorflow. In order to adapt to the size of CIFAR10, I adjusted some parameters in the network. 

- LeNet
- AlexNet
- VGG16
- GoogLeNet
- ResNet50

## Requirements

- python 3.6.5
- tensorflow 1.2.1
- numpy
- CIFAR10 can be download [here][1]. The path to ‘cifar-10-batches-py’ can be specified with the optional parameter ‘--dataset_dir’, which by default is placed in the root directory.
	
## Train

```
# Train by default.
$ python main.py

# Train with optional patameters.
$ python main.py --model_type [LeNet/AlexNet/VGG16/GoogLeNet/ResNet50] --dataset_dir [Path to cifar-10-batches-py]
```

  [1]: https://www.cs.toronto.edu/~kriz/cifar.html

