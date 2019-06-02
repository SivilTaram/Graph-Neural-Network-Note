# 从图(Graph)到图卷积(Graph Convolution): 漫谈图神经网络

*Github Markdown 对 Latex 的支持不好，推荐移步[本人博客](https://www.cnblogs.com/SivilTaram/p/graph_neural_network.html)阅读，之后会同步更新。*

笔者最近看了一些图与图卷积神经网络的论文，深感其强大，但一些Survey或教程默认了读者对图神经网络背景知识的了解，对未学过信号处理的读者不太友好。同时，很多教程只讲是什么，不讲为什么，也没有梳理清楚不同网络结构的区别与设计初衷(Motivation)。

因此，本文试图沿着图神经网络的历史脉络，从最早基于不动点理论的**图神经网络**(Graph Neural Network， GNN)一步步讲到当前用得最火的**图卷积神经网络**(Graph Convolutional Neural Network， GCN)， 期望通过本文带给读者一些灵感与启示。

- 本文的提纲与叙述要点主要参考了2篇图神经网络的Survey，分别是来自IEEE Fellow的*A Comprehensive Survey on Graph Neural Networks*[1] 以及来自清华大学朱文武老师组的*Deep Learning on Graphs: A Survey*[7]， 在这里向两篇Survey的作者表示敬意。
- 同时，本文关于部分图卷积神经网络的**理解**很多都是受到知乎问题[8]高赞答案的启发，非常感谢他们的无私分享！
- 最后，本文还引用了一些来自互联网的生动形象的图片，在这里也向这些图片的作者表示感谢。

## 历史

在开始正文之前，笔者先带大家回顾一下图神经网络的发展历史。不过，因为图神经网络的发展分支非常之多，笔者某些叙述可能并不全面，一家之言仅供各位读者参考：

1. 图神经网络的概念最早在2005年提出。2009年Franco博士在其论文 [2]中定义了图神经网络的理论基础，笔者呆会要讲的第一种图神经网络也是基于这篇论文。
2. 最早的GNN主要解决的还是如分子结构分类等严格意义上的图论问题。但实际上欧式空间(比如像图像 Image)或者是序列(比如像文本 Text)，许多常见场景也都可以转换成图(Graph)，然后就能使用图神经网络技术来建模。
3. 2009年后图神经网络也陆续有一些相关研究，但没有太大波澜。直到2013年，在图信号处理(Graph Signal Processing)的基础上，Bruna(这位是LeCun的学生)在文献 [3]中首次提出图上的基于频域(Spectral-domain)和基于空域(Spatial-domain)的卷积神经网络。
4. 其后至今，学界提出了很多基于空域的图卷积方式，也有不少学者试图通过统一的框架将前人的工作统一起来。而基于频域的工作相对较少，只受到部分学者的青睐。
5. 值得一提的是，图神经网络与图表示学习(Represent Learning for Graph)的发展历程也惊人地相似。2014年，在word2vec [4]的启发下，Perozzi等人提出了DeepWalk [5]，开启了深度学习时代图表示学习的大门。更有趣的是，就在几乎一样的时间，Bordes等人提出了大名鼎鼎的TransE [6]，为知识图谱的分布式表示(Represent Learning for Knowledge Graph)奠定了基础。

## 图神经网络(Graph Neural Network)

首先要澄清一点，除非特别指明，本文中所提到的图均指**图论中的图**(Graph)。它是一种由若干个**结点**(Node)及连接两个结点的**边**(Edge)所构成的图形，用于刻画不同结点之间的关系。下面是一个生动的例子，图片来自 [1]:

![图像与图示例](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-1-image-and-graph.png)

### 状态更新与输出

最早的图神经网络起源于Franco博士的论文[2], 它的理论基础是**不动点**理论。给定一张图 $G$，每个结点都有其自己的特征(feature), 本文中用$\mathbf{x}_v$表示结点v的特征；连接两个结点的边也有自己的特征，本文中用$\mathbf{x}_{(v,u)}$表示结点v与结点u之间边的特征；GNN的学习目标是获得每个结点的图感知的隐藏状态 $\mathbf{h}_v$(state embedding)，这就意味着：对于每个节点，它的隐藏状态包含了来自邻居节点的信息。那么，如何让每个结点都感知到图上其他的结点呢？GNN通过**迭代式更新**所有结点的隐藏状态来实现，在$t+1$时刻，结点$v$的隐藏状态按照如下方式更新：

$$𝐡^{t+1}_𝑣=𝑓(𝐱_𝑣,𝐱_𝑐𝑜[𝑣],𝐡^{t}_𝑛𝑒[𝑣] ,𝐱_𝑛𝑒[𝑣]),
$$

上面这个公式中的 $f$ 就是隐藏状态的**状态更新**函数，在论文中也被称为**局部转移函数**(local transaction function)。公式中的$𝐱_𝑐𝑜[𝑣]$指的是与结点$v$相邻的边的特征，$𝐱_𝑛𝑒[𝑣]$指的是结点$v$的邻居结点的特征，$𝐡^t_𝑛𝑒[𝑣]$则指邻居结点在$t$时刻的隐藏状态。注意 $f$ 是对所有结点都成立的，是一个全局共享的函数。那么怎么把它跟深度学习结合在一起呢？聪明的读者应该想到了，那就是利用神经网络(Neural Network)来拟合这个复杂函数 $f$。值得一提的是，虽然看起来 $f$ 的输入是不定长参数，但在 $f$ 内部我们可以先将不定长的参数通过一定操作变成一个固定的参数，比如说用所有隐藏状态的加和来代表所有隐藏状态。我们举个例子来说明一下：

![更新公式示例](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-2-state-update-function.png)

假设结点$5$为中心结点，其隐藏状态的更新函数如图所示。这个更新公式表达的思想自然又贴切：不断地利用当前时刻邻居结点的隐藏状态作为部分输入来生成下一时刻中心结点的隐藏状态，直到每个结点的隐藏状态变化幅度很小，整个图的信息流动趋于平稳。至此，每个结点都“知晓”了其邻居的信息。状态更新公式仅描述了如何获取每个结点的隐藏状态，除它以外，我们还需要另外一个函数 $g$ 来描述如何适应下游任务。举个例子，给定一个社交网络，一个可能的下游任务是判断各个结点是否为水军账号。

$$𝐨_𝑣=𝑔(𝐡_𝑣,𝐱_𝑣)$$

在原论文中，$g$ 又被称为**局部输出函数**(local output function)，与 $f$ 类似，$g$ 也可以由一个神经网络来表达，它也是一个全局共享的函数。

### 不动点理论

在本节的开头我们就提到了，GNN的理论基础是**不动点**(the fixed point)理论，这里的不动点理论专指**巴拿赫不动点定理**(Banach's Fixed Point Theorem)。首先我们用 $F$ 表示若干个 $f$ 堆叠得到的一个函数，也称为**全局更新**函数，那么图上所有结点的状态更新公式可以写成：

$$𝐇^{𝑡+1}=F(𝐇^𝑡,𝐗)$$

不动点定理指的就是，不论$\mathbf{H}^0$是什么，只要 $F$ 是个**压缩映射**(contraction map)，$\mathbf{H}^{0}$经过不断迭代都会收敛到某一个固定的点，我们称之为不动点。那压缩映射又是什么呢，一张图给你解释得明明白白：

![更新公式示例](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-3-contraction-map.png)

上图的实线箭头就是指映射 $F$, 任意两个点 $x,y$ 在经过 $F$ 这个映射后，分别变成了 $F(x),F(y)$。压缩映射就是指，$𝑑(𝐹(𝑥),𝐹(𝑦))≤𝑐𝑑(𝑥,𝑦), 0≤𝑐<1$。也就是说，经过 $F$ 变换后的新空间一定比原先的空间要小，原先的空间被压缩了。想象这种压缩的过程不断进行，最终就会把原空间中的所有点映射到一个小小的点上，即不动点。

## 参考文献

[1]. A Comprehensive Survey on Graph Neural Networks, https://arxiv.org/abs/1901.00596

[2]. The graph neural network model, https://persagen.com/files/misc/scarselli2009graph.pdf

[3]. Spectral networks and locally connected networks on graphs, https://arxiv.org/abs/1312.6203

[4]. Distributed Representations of Words and Phrases and their Compositionality, http://papers.nips.cc/paper/5021-distributed-representations-of-words-andphrases

[5]. DeepWalk: Online Learning of Social Representations, https://arxiv.org/abs/1403.6652

[6]. Translating Embeddings for Modeling Multi-relational Data, https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data

[7]. Deep Learning on Graphs: A Survey, https://arxiv.org/abs/1812.04202

[8]. 如何理解Graph Convolutional Network（GCN）? https://www.zhihu.com/question/54504471
