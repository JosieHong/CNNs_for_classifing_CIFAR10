# Handwritten digit recognition based on CNN

## Environment

- Python 3.6.5
- Tensorflow 1.8.0

## Network Structure

Input layer: input the original training image ----------------------->(28,28,1)
Conv1: 16 5*5 convolution kernels, the step strides is 1----->(28,28,16)
Pooling1: The convolution kernel size is 2*2, and the step strides is 2---->(14,14,16)
Conv2: 32 5*5 convolution kernels, the step strides is 1----->(14,14,32)
Pooling2: The convolution kernel size is 2*2, and the step strides is 2---->(7,7,32)
Output layer: The output is a 10-dimensional vector

The network structure derived by `CNN_tb.py` is as follows:
![net.png-114.7kB][1]

## Results

```
Step: 0 | train loss: 2.2956 | test accuracy: 0.14
Step: 50 | train loss: 0.3620 | test accuracy: 0.48
Step: 100 | train loss: 0.2210 | test accuracy: 0.61
Step: 150 | train loss: 0.2453 | test accuracy: 0.69
Step: 200 | train loss: 0.1980 | test accuracy: 0.74
Step: 250 | train loss: 0.2198 | test accuracy: 0.77
Step: 300 | train loss: 0.0816 | test accuracy: 0.79
Step: 350 | train loss: 0.0705 | test accuracy: 0.81
Step: 400 | train loss: 0.1075 | test accuracy: 0.83
Step: 450 | train loss: 0.0772 | test accuracy: 0.84
Step: 500 | train loss: 0.1945 | test accuracy: 0.85
Step: 550 | train loss: 0.1190 | test accuracy: 0.86
[7 2 1 0 4 1 4 9 5 9] prediction number
[7 2 1 0 4 1 4 9 5 9] real number
```

The visualization results are shown below. 

![image_1csamfg1c18j61vat1foh1821prc3l.png-64.8kB][2]

Here are some Chinese notes on CNN in `Handwritten-digit-recognition-based-on-CNN/How Convolutional Neural Networks Work/`. 

  [1]: http://static.zybuluo.com/JosieException/g3ov9nqqgh2352csl7kn6i9a/net.png
  [2]: http://static.zybuluo.com/JosieException/j2vhhu8ikeh4de6nt3u1mpj7/image_1csamfg1c18j61vat1foh1821prc3l.png