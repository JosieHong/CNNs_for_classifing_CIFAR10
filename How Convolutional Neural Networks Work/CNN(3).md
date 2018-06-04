# How Convolutional Neural Networks Work（3）

## 神经网络是怎样工作的（3）

> 原视频地址：https://www.youtube.com/watch?v=FmpDIaiMIeA&t=51s
> 
> 文字为个人理解和部分字幕翻译

**目录：**

神经网络是怎样工作的（1）

1. CNN概述

2. Convolution Layer (卷积层)

神经网络是怎样工作的（2）

3. Pooling Layer (池化层)

4. Normalization (标准化)

5. Fully Connected Layer (全连接层)

6. Layers Get Stacked (各层堆叠)

神经网络是怎样工作的（3）

7. Others (其他)


----------

### 7. Others (其他)

这里我们还有一些需要注意的。

[![39.png](https://i.loli.net/2018/06/04/5b154db521165.png)](https://i.loli.net/2018/06/04/5b154db521165.png)

> Q: 这些有魔力的参数来自哪里？
>
> 卷积层的特征部分
> 
> 全连接层的投票权重
> 
> A: 反向传播

另一个小技巧来获得这些神奇的参数，叫做反向传播（Backpropagation）神经网络会自己学习到这些数据，不需要我们知道它们具体的数值也不需要猜测。

[![40.png](https://i.loli.net/2018/06/04/5b154db590c91.png)](https://i.loli.net/2018/06/04/5b154db590c91.png)

反向传播暗含的准则是，用误差值来衡量一个神经网络是否好用。

[![41.png](https://i.loli.net/2018/06/04/5b154f08c03ff.png)](https://i.loli.net/2018/06/04/5b154f08c03ff.png)

如果我们已知我们的输入是X，并且输出的结果为是X的匹配度是0.92，是O的匹配度是0.51，那么误差值就是
（1-0.92）+（1-0.51） = 0.57

[![42.png](https://i.loli.net/2018/06/04/5b154f0801e8e.png)](https://i.loli.net/2018/06/04/5b154f0801e8e.png)

> 梯度下降
> 
> 对于每个像素特征和投票权重，向上或者向下调整一点，再看误差值的变化。

就像在一个斜坡上滚动小球，它不停的左右滚动最终我们可以找到误差最小的位置停下来。

[![43.png](https://i.loli.net/2018/06/04/5b154f08000d8.png)](https://i.loli.net/2018/06/04/5b154f08000d8.png)

> **超级参数**
> 
> **卷积层：**特征部分的数量、特征部分大小
> 	
> **池化层：**窗口大小、窗口步长
> 	
> **全连接层：**神经元个数
	
目前还没有确定的又好又快的方法来确定这些参数，实际上一些先进的神经网络采用结合的方法，这些方法比较好用。

[![44.png](https://i.loli.net/2018/06/04/5b154f064746c.png)](https://i.loli.net/2018/06/04/5b154f064746c.png)

> 体系结构
> 
> 各种层的数量？
> 
> 它们的怎样组合？

[![45.png](https://i.loli.net/2018/06/04/5b154f077982c.png)](https://i.loli.net/2018/06/04/5b154f077982c.png)

> 不仅用于图片
> 
> 任何二维（或者三维）的数据
> 
> 关系紧密的数据比关系疏远的数据更适用

例如：

[![46.png](https://i.loli.net/2018/06/04/5b154f059e8a9.png)](https://i.loli.net/2018/06/04/5b154f059e8a9.png)

[![47.png](https://i.loli.net/2018/06/04/5b154f06868f6.png)](https://i.loli.net/2018/06/04/5b154f06868f6.png)

[![48.png](https://i.loli.net/2018/06/04/5b154f0673e36.png)](https://i.loli.net/2018/06/04/5b154f0673e36.png)

[![49.png](https://i.loli.net/2018/06/04/5b154f07d04c3.png)](https://i.loli.net/2018/06/04/5b154f07d04c3.png)

> 局限性
> 
> 卷积神经网络只适用于“空间的”数据组。
> 
> 如果数据不能组成图片那样的形式，卷积神经网络就不那么好用了。

例如：

[![50.png](https://i.loli.net/2018/06/04/5b154f08be313.png)](https://i.loli.net/2018/06/04/5b154f08be313.png)

[![51.png](https://i.loli.net/2018/06/04/5b155136515d1.png)](https://i.loli.net/2018/06/04/5b155136515d1.png)

> 好用的准则
> 
> 如果数据在交换各列之后仍然是可用的，那么它不适用于卷积神经网络。

[![52.png](https://i.loli.net/2018/06/04/5b15513618346.png)](https://i.loli.net/2018/06/04/5b15513618346.png)






