# CNNs_for_classifing_CIFAR10

Implement of CNNs for classifing on CIFAR10 with tensorflow(CPU). In order to adapt to the size of CIFAR10, I adjusted some parameters in the network. And it's easy to fit it taining on other dataset. 

- LeNet
- AlexNet
- VGG16
- GoogLeNet
- ResNet50

## Requirements

- python 3.6.3
- tensorflow 1.13.1
- numpy 1.16.3
- CIFAR10 can be download [here][1]. The path to ‘cifar-10-batches-py’ can be specified with the optional parameter ‘--dataset_dir’, which by default is placed in the root directory.

## Train and Test

Here I only iterate 20 epoches (10000 steps), you can increase the number of iterations by using the last trained model to achieve higher accuracy. Besides, you can also change `learning rate` and `steps` in `main.py`.

```
# Train and test by default.
$ python main.py

# Train with optional patameters and test.
$ python main.py	--model_type	[LeNet/AlexNet/VGG16/GoogLeNet/ResNet50] 
			--dataset_dir	[Path to cifar-10-batches-py] 
			--model_dir	[A .ckpt file of pretrained model or A folder for saving model] 
```

## Use GPU

- CUDA 8.0.61
- CUDNN 5.1
- tensorflow_gpu 1.2.0

```
# Chose GPU to use
$ CUDA_VISIBLE_DEVICES=0 python main.py (optional patameters...)
```

## Logs

```
$ tensorboard --logdir=/logs
```

  [1]: https://www.cs.toronto.edu/~kriz/cifar.html

