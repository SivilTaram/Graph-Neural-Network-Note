# 从图(Graph)到图卷积(Graph Convolution): 漫谈图神经网络

*Github Markdown 对 Latex 的支持不好，推荐移步[本人博客](https://www.cnblogs.com/SivilTaram/p/graph_neural_network.html)阅读，之后会同步更新。*

笔者最近看了一些图与图卷积神经网络的论文，深感其强大，但一些Survey或教程默认了读者对图神经网络背景知识的了解，对未学过信号处理的读者不太友好。同时，很多教程只讲是什么，不讲为什么，也没有梳理清楚不同网络结构的区别与设计初衷(Motivation)。

因此，本文试图沿着图神经网络的历史脉络，从最早基于不动点理论的**图神经网络**(Graph Neural Network， GNN)一步步讲到当前用得最火的**图卷积神经网络**(Graph Convolutional Neural Network， GCN)， 期望通过本文带给读者一些灵感与启示。

- 本文的提纲与叙述要点主要参考了2篇图神经网络的Survey，分别是来自IEEE Fellow的*A Comprehensive Survey on Graph Neural Networks*[1] 以及来自清华大学朱文武老师组的*Deep Learning on Graphs: A Survey*[7]， 在这里向两篇Survey的作者表示敬意。
- 同时，本文关于部分图卷积神经网络的**理解**很多都是受到知乎问题[8]高赞答案的启发，非常感谢他们的无私分享！
- 最后，本文还引用了一些来自互联网的生动形象的图片，在这里也向这些图片的作者表示感谢。本文中未注明出处的图片均为笔者制作，如需转载或引用请联系本人。

## 历史脉络

在开始正文之前，笔者先带大家回顾一下图神经网络的发展历史。不过，因为图神经网络的发展分支非常之多，笔者某些叙述可能并不全面，一家之言仅供各位读者参考：

1. 图神经网络的概念最早在2005年提出。2009年Franco博士在其论文 [2]中定义了图神经网络的理论基础，笔者呆会要讲的第一种图神经网络也是基于这篇论文。
2. 最早的GNN主要解决的还是如分子结构分类等严格意义上的图论问题。但实际上欧式空间(比如像图像 Image)或者是序列(比如像文本 Text)，许多常见场景也都可以转换成图(Graph)，然后就能使用图神经网络技术来建模。
3. 2009年后图神经网络也陆续有一些相关研究，但没有太大波澜。直到2013年，在图信号处理(Graph Signal Processing)的基础上，Bruna(这位是LeCun的学生)在文献 [3]中首次提出图上的基于频域(Spectral-domain)和基于空域(Spatial-domain)的卷积神经网络。
4. 其后至今，学界提出了很多基于空域的图卷积方式，也有不少学者试图通过统一的框架将前人的工作统一起来。而基于频域的工作相对较少，只受到部分学者的青睐。
5. 值得一提的是，图神经网络与图表示学习(Represent Learning for Graph)的发展历程也惊人地相似。2014年，在word2vec [4]的启发下，Perozzi等人提出了DeepWalk [5]，开启了深度学习时代图表示学习的大门。更有趣的是，就在几乎一样的时间，Bordes等人提出了大名鼎鼎的TransE [6]，为知识图谱的分布式表示(Represent Learning for Knowledge Graph)奠定了基础。

## 图神经网络(Graph Neural Network)

首先要澄清一点，除非特别指明，本文中所提到的图均指**图论中的图**(Graph)。它是一种由若干个**结点**(Node)及连接两个结点的**边**(Edge)所构成的图形，用于刻画不同结点之间的关系。下面是一个生动的例子，图片来自论文[7]:

![图像与图示例](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-1-image-and-graph.png)

### 状态更新与输出

最早的图神经网络起源于Franco博士的论文[2], 它的理论基础是**不动点**理论。给定一张图 $G$，每个结点都有其自己的特征(feature), 本文中用$\mathbf{x}_v$表示结点v的特征；连接两个结点的边也有自己的特征，本文中用$\mathbf{x}_{(v,u)}$表示结点v与结点u之间边的特征；GNN的学习目标是获得每个结点的图感知的隐藏状态 $\mathbf{h}_v$(state embedding)，这就意味着：对于每个节点，它的隐藏状态包含了来自邻居节点的信息。那么，如何让每个结点都感知到图上其他的结点呢？GNN通过**迭代式更新**所有结点的隐藏状态来实现，在$t+1$时刻，结点$v$的隐藏状态按照如下方式更新：

$$𝐡^{t+1}_𝑣=𝑓(𝐱_𝑣,𝐱_𝑐𝑜[𝑣],𝐡^{t}_𝑛𝑒[𝑣] ,𝐱_𝑛𝑒[𝑣]),
$$

上面这个公式中的 $f$ 就是隐藏状态的**状态更新**函数，在论文中也被称为**局部转移函数**(local transaction function)。公式中的$𝐱_𝑐𝑜[𝑣]$指的是与结点$v$相邻的边的特征，$𝐱_𝑛𝑒[𝑣]$指的是结点$v$的邻居结点的特征，$𝐡^t_𝑛𝑒[𝑣]$则指邻居结点在$t$时刻的隐藏状态。注意 $f$ 是对所有结点都成立的，是一个全局共享的函数。那么怎么把它跟深度学习结合在一起呢？聪明的读者应该想到了，那就是利用神经网络(Neural Network)来拟合这个复杂函数 $f$。值得一提的是，虽然看起来 $f$ 的输入是不定长参数，但在 $f$ 内部我们可以先将不定长的参数通过一定操作变成一个固定的参数，比如说用所有隐藏状态的加和来代表所有隐藏状态。我们举个例子来说明一下：

![更新公式示例](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-2-state-update-function.png)

假设结点$5$为中心结点，其隐藏状态的更新函数如图所示。这个更新公式表达的思想自然又贴切：不断地利用当前时刻邻居结点的隐藏状态作为部分输入来生成下一时刻中心结点的隐藏状态，直到每个结点的隐藏状态变化幅度很小，整个图的信息流动趋于平稳。至此，每个结点都“知晓”了其邻居的信息。状态更新公式仅描述了如何获取每个结点的隐藏状态，除它以外，我们还需要另外一个函数 $g$ 来描述如何适应下游任务。举个例子，给定一个社交网络，一个可能的下游任务是判断各个结点是否为水军账号。

$$𝐨_𝑣=𝑔(𝐡_𝑣,𝐱_𝑣)$$

在原论文中，$g$ 又被称为**局部输出函数**(local output function)，与 $f$ 类似，$g$ 也可以由一个神经网络来表达，它也是一个全局共享的函数。那么，整个流程可以用下面这张图表达：

![更新公式示例](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-4-state-flow.png)

仔细观察两个时刻之间的连线，它与图的连线密切相关。比如说在 $T_1$ 时刻，结点 1 的状态接受来自结点 3 的上一时刻的隐藏状态，因为结点 1 与结点 3相邻。直到 $T_n$ 时刻，各个结点隐藏状态收敛，每个结点后面接一个 $g$ 即可得到该结点的输出 $\mathbf{o}$。

对于不同的图来说，收敛的时刻可能不同，因为收敛是通过两个时刻$p$-范数的差值是否小于某个阈值 $\epsilon$来判定的，比如：

$$||\mathbf{H}^{t+1}||_{2}-||\mathbf{H}^{t}||_{2}<\epsilon$$

### 实例:化合物分类

下面让我们举个实例来说明图神经网络是如何应用在实际场景中的，这个例子来源于论文[2]。假设我们现在有这样一个任务，给定一个环烃化合物的分子结构(包括原子类型，原子键等)，模型学习的目标是判断其是否有害。这是一个典型的二分类问题，一个训练样本如下图所示：

![化合物分子结构](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-6-gnn-example.png)

由于化合物的分类实际上需要对整个图进行分类，在论文中，作者将化合物的**根结点**的表示作为整个图的表示，如图上红色的结点所示。Atom feature 中包括了每个原子的类型(Oxygen, 氧原子)、原子自身的属性(Atom Properties)、化合物的一些特征(Global Properties)等。把每个原子看作图中的结点，原子键视作边，一个分子(Molecule)就可以看作一张图。在不断迭代得到根结点氧原子收敛的隐藏状态后，在上面接一个前馈神经网络作为输出层(即$g$函数)，就可以对整个化合物进行二分类了。
> 当然，在同构图上根据策略选择同一个根结点对结果也非常重要。但在这里我们不关注这部分细节，感兴趣的读者可以阅读原文。

### 不动点理论

在本节的开头我们就提到了，GNN的理论基础是**不动点**(the fixed point)理论，这里的不动点理论专指**巴拿赫不动点定理**(Banach's Fixed Point Theorem)。首先我们用 $F$ 表示若干个 $f$ 堆叠得到的一个函数，也称为**全局更新**函数，那么图上所有结点的状态更新公式可以写成：

$$𝐇^{𝑡+1}=F(𝐇^𝑡,𝐗)$$

不动点定理指的就是，不论$\mathbf{H}^0$是什么，只要 $F$ 是个**压缩映射**(contraction map)，$\mathbf{H}^{0}$经过不断迭代都会收敛到某一个固定的点，我们称之为不动点。那压缩映射又是什么呢，一张图可以解释得明明白白：

![更新公式示例](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-3-contraction-map.png)

上图的实线箭头就是指映射 $F$, 任意两个点 $x,y$ 在经过 $F$ 这个映射后，分别变成了 $F(x),F(y)$。压缩映射就是指，$𝑑(𝐹(𝑥),𝐹(𝑦))≤𝑐𝑑(𝑥,𝑦), 0≤𝑐<1$。也就是说，经过 $F$ 变换后的新空间一定比原先的空间要小，原先的空间被压缩了。想象这种压缩的过程不断进行，最终就会把原空间中的所有点映射到一个点上。

那么肯定会有读者心存疑问，既然 $f$ 是由神经网络实现的，我们该如何实现它才能保证它是一个压缩映射呢？我们下面来谈谈 $f$ 的具体实现。


### 具体实现

在具体实现中， $f$ 其实通过一个简单的**前馈神经网络**(Feed-forward Neural Network)即可实现。比如说，一种实现方法可以是把每个邻居结点的特征、隐藏状态、每条相连边的特征以及结点本身的特征简单拼接在一起，在经过前馈神经网络后做一次简单的加和。

$$𝐡_𝑣^{𝑡+1}=𝑓(𝐱_𝑣,𝐱_𝑐𝑜[𝑣] ,𝐡^t_𝑛𝑒[𝑣] ,𝐱_𝑛𝑒[𝑣])$$
$$
=\sum_{𝑢∈𝑛𝑒[𝑣]} FNN([𝐱_𝑣;𝐱_{(𝑢,𝑣)};𝐡_𝑢^𝑡;𝐱_𝑢])$$

那我们如何保证 $f$ 是个压缩映射呢，其实是通过限制 $f$ 对 $\mathbf{H}$ 的偏导数矩阵的大小，这是通过一个对**雅可比矩阵**(Jacobian Matrix)的**惩罚项**(Penalty)来实现的。在代数中，有一个定理是: $f$ 为压缩映射的等价条件是 $f$ 的梯度/导数要小于1。这个等价定理可以从压缩映射的形式化定义导出，我们这里使用 $||x||$ 表示 $x$ 在空间中的**范数**(norm)。范数是一个标量，它是向量的长度或者模，$||x||$ 是 $x$ 在有限空间中坐标的连续函数。这里把 $x$ 简化成1维的，坐标之间的差值可以看作向量在空间中的距离，根据压缩映射的定义，可以导出：

$$||F(x)-F(y)||{\leq}c||x-y||, 0\ {\leq}c<1$$
$$\frac{||F(x)-F(y)||}{||x-y||}{\leq}c$$
$$\frac{||F(x)-F(x-{\Delta}x)||}{||{\Delta}x||}{\leq}c$$
$$||F'(x)||=||\frac{{\partial}F(x)}{{\partial}x}||{\leq}c$$

推广一下，即得到雅可比矩阵的罚项需要满足其范数小于等于$c$等价于压缩映射的条件。根据拉格朗日乘子法，将有约束问题变成带罚项的无约束优化问题，训练的目标可表示成如下形式：

$$J = Loss + \lambda \cdot \max({\frac{||{\partial}FNN||}{||{\partial}\mathbf{h}||}}−c,0), c\in(0,1)$$

其中$\lambda$是超参数，与其相乘的项即为雅可比矩阵的罚项。

### 模型学习

上面我们花一定的篇幅搞懂了如何让 $f$ 接近压缩映射，下面我们来具体叙述一下图神经网络中的损失 $Loss$ 是如何定义，以及模型是如何学习的。

仍然以社交网络举例，虽然每个结点都会有隐藏状态以及输出，但并不是每个结点都会有**监督信号**(Supervision)。比如说，社交网络中只有部分用户被明确标记了是否为水军账号，这就构成了一个典型的结点二分类问题。

那么很自然地，模型的损失即通过这些有监督信号的结点得到。假设监督结点一共有 $p$ 个，模型损失可以形式化为：

$$L𝑜𝑠𝑠=∑_{𝑖=1}^𝑝{(𝐭_𝑖−𝐨_𝑖)}$$

那么，模型如何学习呢？根据**前向传播计算损失**的过程，不难推出**反向传播计算梯度**的过程。在前向传播中，模型：
1. 调用 $f$ 若干次，比如 $T_n$次，直到 $\mathbf{h}^{T_{n}}_v$ 收敛。
2. 此时每个结点的隐藏状态接近不动点的解。
3. 对于有监督信号的结点，将其隐藏状态通过 $g$ 得到输出，进而算出模型的损失。

根据上面的过程，在反向传播时，我们可以直接求出 $f$ 和 $g$ 对最终的隐藏状态  $\mathbf{h}^{T_{n}}_v$ 的梯度。然而，因为模型递归调用了 $f$ 若干次，为计算 $f$ 和 $g$ 对最初的隐藏状态 $\mathbf{h}_v^0$ 的梯度，我们需要同样递归式/迭代式地计算 $T_n$ 次梯度。最终得到的梯度即为 $f$ 和 $g$ 对 $\mathbf{h}_v^0$ 的梯度，然后该梯度用于更新模型的参数。这个算法就是 Almeida-Pineda 算法[9]。
<!-- 之所以要求 $f$ 为压缩映射，也是因为只有 $f$ 为压缩映射时，AP 才能得到一个收敛的梯度。 -->

### GNN与RNN

相信熟悉 RNN/LSTM/GRU 等循环神经网络的同学看到这里会有一点小困惑，因为图神经网络不论是前向传播的方式，还是反向传播的优化算法，与循环神经网络都有点相像。这并不是你的错觉，实际上，图神经网络与到循环神经网络确实很相似。为了清楚地显示出它们之间的不同，我们用一张图片来解释这两者设计上的不同：

![GNN与RNN的区别](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-5-gnn-rnn.png)

假设在GNN中存在三个结点$x_1$,$x_2$,$x_3$，相应地，在RNN中有一个序列$(x_1,x_2,x_3)$。笔者认为，GNN与RNN的区别主要在于4点：
- GNN的基础理论是不动点理论，这就意味着GNN沿时间展开的长度是动态的，是**根据收敛条件**确定的，而RNN沿时间展开的长度就等于**序列本身的长度**。
- GNN每次时间步的输入都是所有结点 $v$ 的特征，而RNN每次时间步的输入是该时刻对应的输入。同时，时间步之间的信息流也不相同，前者由边决定，后者则由序列的读入顺序决定。
- GNN采用 AP 算法反向传播优化，而RNN使用**BPTT**(Back Propogation Through Time)优化。前者对收敛性有要求，而后者对收敛性是没有要求的。
- GNN循环调用 $f$ 的目标是得到每个结点稳定的隐藏状态，所以只有在隐藏状态收敛后才能输出；而RNN的每个时间步上都可以输出，比如语言模型。

不过鉴于初代GNN与RNN结构上的相似性，一些文章中也喜欢把它称之为 Recurrent-based GNN，也有一些文章会把它归纳到 Recurrent-based GCN中。之后读者在了解 GCN后会理解为什么人们要如此命名。

### 小结

初代GNN，也就是基于循环结构的图神经网络的核心是不动点理论。它的核心观点是**通过结点信息的传播使整张图达到收敛，在其基础上再进行预测**。

## 门控图神经网络(Gated Graph Neural Network)

我们上面细致比较了GNN与RNN，可以发现它们有诸多相通之处。那么，我们能不能直接用类似RNN的方法来定义GNN呢? 于是乎，**门控图神经网络**(Gated Graph Neural Network) [10]就出现了。虽然在这里它们看起来类似，但实际上，它们的区别非常大，其中最核心的不同即是**门控神经网络不以不动点理论为基础**。这意味着：$f$ 不再需要是一个压缩映射；迭代不需要到收敛才能输出，可以迭代固定步长；优化算法也从 AP 算法转向 BPTT。



## 参考文献


[1]. A Comprehensive Survey on Graph Neural Networks, https://arxiv.org/abs/1901.00596

[2]. The graph neural network model, https://persagen.com/files/misc/scarselli2009graph.pdf

[3]. Spectral networks and locally connected networks on graphs, https://arxiv.org/abs/1312.6203

[4]. Distributed Representations of Words and Phrases and their Compositionality, http://papers.nips.cc/paper/5021-distributed-representations-of-words-andphrases

[5]. DeepWalk: Online Learning of Social Representations, https://arxiv.org/abs/1403.6652

[6]. Translating Embeddings for Modeling Multi-relational Data, https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data

[7]. Deep Learning on Graphs: A Survey, https://arxiv.org/abs/1812.04202

[8]. 如何理解Graph Convolutional Network（GCN）? https://www.zhihu.com/question/54504471

[9]. Almeida–Pineda recurrent backpropagation, https://www.wikiwand.com/en/Almeida%E2%80%93Pineda_recurrent_backpropagation

[10]. Gated graph sequence neural networks, https://arxiv.org/abs/1511.05493