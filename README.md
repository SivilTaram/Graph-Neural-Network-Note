# 从图(Graph)到图卷积(Graph Convolution): 漫谈图神经网络

笔者最近看了一些图与图卷积神经网络的论文，深感其强大，但一些Survey或教程默认了读者对图神经网络背景知识的了解，对未学过信号处理的读者不太友好。同时，很多教程只讲是什么，不讲为什么，也没有梳理清楚不同网络结构的区别与设计初衷(Motivation)。因此，本文试图沿着图神经网络的历史脉络，从最早基于不动点理论的**图神经网络**(Graph Neural Network， GNN)一步步讲到当前用得最火的**图卷积神经网络**(Graph Convolutional Neural Network， GCN)， 期望通过本文带给读者一些灵感与启示。

本文的提纲与叙述要点主要参考了2篇图神经网络的Survey，分别是来自IEEE Fellow的*A Comprehensive Survey on Graph Neural Networks* 以及来自清华大学朱文武老师组的*Deep Learning on Graphs: A Survey*， 在这里向两篇Survey的作者表示敬意。同时，在本文的部分内容还引用了一些来自互联网的图片，这些图片都非常生动形象，在这里也向这些图片的作者表示感谢。最后，本文关于部分图卷积神经网络的描述源于知乎问题[如何理解GCN](https://www.zhihu.com/question/54504471
)中高赞答案中的前两位，向他们表示致敬！

## 历史

在开始正文之前，笔者先带大家回顾一下图神经网络的发展历史。不过，因为图神经网络的发展分支非常之多，笔者某些叙述可能并不全面，一家之言仅供各位读者参考：
1. 图神经网络的概念最早在2005年提出。2009年Franco博士在[The graph neural network model](https://persagen.com/files/misc/scarselli2009graph.pdf)一文中定义了图神经网络的理论基础，笔者呆会要讲的第一种图神经网络也是基于这篇论文。
2. 最早的GNN主要解决的还是如分子结构分类等严格意义上的图论问题。但实际上欧式空间(比如像图像 Image)或者是序列(比如像文本 Text)，许多常见场景也都可以转换成图(Graph)，然后就能使用图神经网络技术来建模。
3. 2009年后图神经网络也陆续有一些相关研究，但没有太大波澜。直到2013年，在图信号处理(Graph Signal Processing)的基础上，Bruna(应该是提出卷积神经网络的LeCun的学生)在[Spectral networks and locally connected networks on graphs](https://arxiv.org/abs/1312.6203)中首次提出图上的基于频域(Spectral-domain)和基于空域(Spatial-domain)的卷积神经网络。
4. 其后至今，学界提出了很多基于空域的图卷积方式，也有不少学者试图通过统一的框架将前人的工作统一起来。而基于频域的工作相对较少，只受到部分学者的青睐。
5. 值得一提的是，图神经网络与图表示学习(Represent Learning for Graph)的发展历程也惊人地相似。2014年，在[word2vec](http://papers.nips.cc/paper/5021-distributed-representations-of-words-andphrases)的启发下，Perozzi等人提出了[DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)，开启了深度学习时代图表示学习的大门。更有趣的是，就在几乎一样的时间，Bordes等人提出了大名鼎鼎的TransE，即[Translating Embeddings for Modeling Multi-relational Data](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data)，为知识图谱的分布式表示(Represent Learning for Knowledge Graph)奠定了基础。


## 图神经网络(Graph Neural Network)

首先要澄清一点，除非特别指明，本文中所提到的图均指**图论中的图**(Graph)。它是一种由若干个**结点**(Node)及连接两个结点的**边**(Edge)所构成的图形，用于刻画不同结点之间的关系。下面是一个生动的例子，图片源自[这里](https://arxiv.org/abs/1812.04202v1):