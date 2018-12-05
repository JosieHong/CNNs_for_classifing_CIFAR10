# About_CNN

Here is my implement of LeNet and AlexNet for classifing on CIFAR10. In order to adapt to the size of CIFAR10, I adjusted some parameters in the network. 

**It is still being updated...**

- LeNet
- AlexNet
- [Todo]VGG
- [Todo]GoogLeNet
- [Todo]ResNet

## Requirements

- Python 3.6.5
- Tensorflow 1.2.1
- numpy
- CIFAR10
	CIFAR10 can be download [here][1]. The path to ‘cifar-10-batches-py’ can be specified with the optional parameter ‘--dataset_dir’, which by default is placed in the root directory.
	
## Train

```
# Train by default.
python main.py

# Train with optional patameters.
python main.py --model_type \[LeNet/AlexNet] --dataset_dir \[Path to cifar-10-batches-py]
```

## LeNet

![image_1ctnk355mrui1ug15sv1p8p14tap.png-342.6kB][2]

## AlenNet

![image_1ctv435tk1u9d1jru1o5gap3m6q9.png-508.9kB][3]

  [1]: https://www.cs.toronto.edu/~kriz/cifar.html
  [2]: http://static.zybuluo.com/JosieException/236ptteo6xg17pn17rrulplx/image_1ctnk355mrui1ug15sv1p8p14tap.png
  [3]: http://static.zybuluo.com/JosieException/lhlds2qb62pihhsn8qw8np6b/image_1ctv435tk1u9d1jru1o5gap3m6q9.png

