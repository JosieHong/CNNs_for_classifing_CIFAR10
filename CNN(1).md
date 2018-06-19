# How Convolutional Neural Networks Work（1）

## 神经网络是怎样工作的（1）

*原视频地址：https://www.youtube.com/watch?v=FmpDIaiMIeA&t=51s*

*文字为个人理解和部分字幕翻译*

----------

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

### 1.CNN概述

[![1.png](https://i.loli.net/2018/06/04/5b152e3a1fe87.png)](https://i.loli.net/2018/06/04/5b152e3a1fe87.png)

[![2.png](https://i.loli.net/2018/06/04/5b152e4068a97.png)](https://i.loli.net/2018/06/04/5b152e4068a97.png)

>卷积神经网络被认为是可扩展的无监督学习框架的表达

正如上图所展示的，如果你输入一些人脸的图片，在卷积神经网络的多层结构中，第一二层可以识别出一些边和深色的点，但到第三层几乎就可以识别出人脸了，输入汽车的照片也是这样。

[![3.png](https://i.loli.net/2018/06/04/5b152e3de38fd.png)](https://i.loli.net/2018/06/04/5b152e3de38fd.png)

>用加强深度学习玩雅达利（美国一家电脑游戏机厂商）的游戏

CNN甚至可以通过学习这些屏幕上的像素的模式，来学到在不同模式下最优的行为。在学习视频游戏这方面，CNN在某种情况下可以做的远比人类要好。

[![4.png](https://i.loli.net/2018/06/04/5b152e4046445.png)](https://i.loli.net/2018/06/04/5b152e4046445.png)

>机器人通过“观看”互联网上无约束的视频学习操作方法

如果你有一对神经网络，一个学习视频中的物体，一个学习物体的控制方式，它们组合起来，机器人就可以在油管上学习做饭了。

[![5.png](https://i.loli.net/2018/06/04/5b152e3d99f0e.png)](https://i.loli.net/2018/06/04/5b152e3d99f0e.png)

>一个小型卷积神经网络：X 和 O
>
>展示怎样判断一个图片是X还是O

所以毫无疑问，CNN的力量是很强大的。甚至有时候谈及CNN，就像说起一种魔法一样，但它没有什么魔法，CNN是建立在一些简单思想基础上，把这些简单的思想巧妙的运用起来。

CNN的输入是一张图片的二维像素矩阵，你可以想象它是一个表格，表格的每个方框都是暗的或亮的，通过“看”它，CNN会判断一张图片是X还是O。

[![6.png](https://i.loli.net/2018/06/04/5b152e3c33fef.png)](https://i.loli.net/2018/06/04/5b152e3c33fef.png)

这里我们看到黑底白字写着的X和O，我们定义它们分别是X和O。

[![7.png](https://i.loli.net/2018/06/04/5b152e3e0f7ff.png)](https://i.loli.net/2018/06/04/5b152e3e0f7ff.png)

有迷惑性的是每一次的X并不总是相同的，O也是这样，可以被移位、放大缩小、选择或者加粗变细，但我们每一次都需要分辨出X和O。

[![8.png](https://i.loli.net/2018/06/04/5b152e3c9fc14.png)](https://i.loli.net/2018/06/04/5b152e3c9fc14.png)

现在困难的点就在于，对于我们判断两张图片是否相同是很直接的，我们甚至不需要过多的思考，但对于计算机这就很难。
计算机看到的是这样一堆数字化的二维像素表格，1和-1。1代表亮的像素点，-1代表暗的像素点。可以做的只是一个一个比较这些像素点是否相同。

[![9.png](https://i.loli.net/2018/06/04/5b152e3ddd804.png)](https://i.loli.net/2018/06/04/5b152e3ddd804.png)

在计算机看来它们有大部分像素点相同，但还是有部分不同的，所以计算机可能无法确定它们是不是相同的。

[![10.png](https://i.loli.net/2018/06/04/5b152e3ef405e.png)](https://i.loli.net/2018/06/04/5b152e3ef405e.png)

因为计算机十分“表面的”的比较方法，所以不能确定它们是否是相同的。

### 2. Convolution Layer (卷积层)

[![11.png](https://i.loli.net/2018/06/04/5b1532d102ef6.png)](https://i.loli.net/2018/06/04/5b1532d102ef6.png)

CNN的一个小技巧就是，比较一部分一部分的内容而不是整张图片。把图片分成小块或者特征部分，就更容易判断两个事物是否是相似的。

[![12.png](https://i.loli.net/2018/06/04/5b1532d163bd9.png)](https://i.loli.net/2018/06/04/5b1532d163bd9.png)

所以对于这个例子，这些特征部分只是3X3的像素矩阵。最左边的是从左上到右下的斜线，最右边的是和它方向相反的斜线，中间是一个小X。它们都是整张图片的一部分。

[![13.png](https://i.loli.net/2018/06/04/5b1532d194a72.png)](https://i.loli.net/2018/06/04/5b1532d194a72.png)

可以看出，如果你把正确的部分放在原图中正确的位置，都恰好可以匹配的上。现在我们划分好了这些部分，我们要考虑更深的一些东西——匹配的数学方法。

[![14.png](https://i.loli.net/2018/06/04/5b1532d2c9a46.png)](https://i.loli.net/2018/06/04/5b1532d2c9a46.png)

[![15.png](https://i.loli.net/2018/06/04/5b1532d617cbb.png)](https://i.loli.net/2018/06/04/5b1532d617cbb.png)

>**过滤：匹配的数学方法**
>
>1. 将碎片排列起来
>
>2. 将碎片中的像素点与原图中对应的像素点相乘
>
>3. 每个相乘的结果相加
>
>4. 再除以碎片像素点的总数

[![16.png](https://i.loli.net/2018/06/04/5b1532d703fe9.png)](https://i.loli.net/2018/06/04/5b1532d703fe9.png)

首先我们把对应的像素点分别相乘，因为这两幅图是相同的，所以相乘的结果总是1。

[![17.png](https://i.loli.net/2018/06/04/5b1532d8e8659.png)](https://i.loli.net/2018/06/04/5b1532d8e8659.png)

把所有相乘的结果相加，再除以总像素点数9，得到的结果为1，所以我们可以把匹配的结果写在这个矩阵里。这就是Filtering（过滤）。

[![18.png](https://i.loli.net/2018/06/04/5b1532e281cf0.png)](https://i.loli.net/2018/06/04/5b1532e281cf0.png)

接下来我们把这个特征部分换一个位置继续匹配，这时得到的结果是0.55，所以我们可以把0.55写在这个位置。通过移动这个特征部分，与原图中不同的部分进行匹配，我们可以得到不同部分的匹配值，这样可以得到一个特征分布图。

[![19.png](https://i.loli.net/2018/06/04/5b1532e2827f9.png)](https://i.loli.net/2018/06/04/5b1532e2827f9.png)

仔细看我们就可以看出，这个特征分布在原图从左上到右下的斜线范围内，这与原图中X的斜线部分是匹配的，数值比较高的1.00和0.77表示这个特征在这些位置的匹配程度更高。

[![20.png](https://i.loli.net/2018/06/04/5b1532e3422af.png)](https://i.loli.net/2018/06/04/5b1532e3422af.png)

我们会对每一个可能的特征进行尝试，得到它们在原图中分布的情况。

[![21.png](https://i.loli.net/2018/06/04/5b1537b637701.png)](https://i.loli.net/2018/06/04/5b1537b637701.png)

这些就表现为对图片的卷积，通过许多的特征部分，创造出一堆特征分布图，这被叫做卷积层。


