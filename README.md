# 从图(Graph)到图卷积(Graph Convolution): 漫谈图神经网络

*Github Markdown 对 Latex 的支持不好，推荐移步[本人博客](https://www.cnblogs.com/SivilTaram/p/graph_neural_network_1.html)阅读，之后会同步更新。*

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

### GNN的局限

初代GNN，也就是基于循环结构的图神经网络的核心是不动点理论。它的核心观点是**通过结点信息的传播使整张图达到收敛，在其基础上再进行预测**。收敛作为GNN的内核，同样局限了其更广泛的使用，其中最突出的是两个问题：
- GNN只将边作为一种传播手段，但并未区分不同边的功能。虽然我们可以在特征构造阶段($\mathbf{x}_{(u,v)}$)为不同类型的边赋予不同的特征，但相比于其他输入，边对结点隐藏状态的影响实在有限。并且GNN没有为边设置独立的可学习参数，也就意味着无法通过模型学习到边的某些特性。
- 如果把GNN应用在*图表示*的场景中，使用不动点理论并不合适。这主要是因为基于不动点的收敛会导致结点之间的隐藏状态间存在较多信息共享，从而导致结点的状态太**过光滑**(Over Smooth)，并且属于结点自身的特征**信息匮乏**(Less Informative)。

下面这张来自维基百科[13]的图可以形象地解释什么是 Over Smooth，其中我们把整个布局视作一张图，每个像素点与其上下左右以及斜上下左右8个像素点相邻，这决定了信息在图上的流动路径。初始时，蓝色表示没有信息量，如果用向量的概念表达即为空向量；绿色，黄色与红色各自有一部分信息量，表达为非空的特征向量。在图上，信息主要从三块有明显特征的区域向其邻接的像素点流动。一开始**不同像素点的区分非常明显**，但在向不动点过渡的过程中，所有像素点都取向一致，最终整个系统形成均匀分布。这样，虽然每个像素点都感知到了全局的信息，**但我们无法根据它们最终的隐藏状态区分它们**。比如说，根据最终的状态，我们是无法得知哪些像素点最开始时在绿色区域。

![OverSmooth](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-9-over-smooth.gif)

在这里笔者再多说几句。事实上，上面这个图与GNN中的信息流动并不完全等价。从笔者来看，如果我们用物理模型来描述它，上面这个图代表的是初始时有3个热源在散发热量，而后就让它们自由演化；但实际上，GNN在每个时间步都会将结点的特征作为输入来更新隐藏状态，这就好像是放置了若干个永远不灭的热源，热源之间会有互相干扰，但最终不会完全一致。

## 门控图神经网络(Gated Graph Neural Network)

我们上面细致比较了GNN与RNN，可以发现它们有诸多相通之处。那么，我们能不能直接用类似RNN的方法来定义GNN呢? 于是乎，**门控图神经网络**(Gated Graph Neural Network, GGNN) [10]就出现了。虽然在这里它们看起来类似，但实际上，它们的区别非常大，其中最核心的不同即是**门控神经网络不以不动点理论为基础**。这意味着：$f$ 不再需要是一个压缩映射；迭代不需要到收敛才能输出，可以迭代固定步长；优化算法也从 AP 算法转向 BPTT。

### 状态更新

与图神经网络定义的范式一致，GGNN也有两个过程：状态更新与输出。相比GNN而言，它主要的区别来源于状态更新阶段。具体地，GGNN参考了GRU的设计，把邻居结点的信息视作输入，结点本身的状态视作隐藏状态，其状态更新函数如下:

$$\mathbf{h}^{t+1}_v=\text{GRU}(\mathbf{h}^{t}_v,\sum_{𝑢∈𝑛𝑒[𝑣]} \mathbf{W}_{edge}𝐡_𝑢^𝑡)$$

如果读者对GRU的更新公式熟悉的话，对上式应该很好理解。仔细观察上面这个公式，除了GRU式的设计外，GGNN还针对不同类型的边引入了可学习的参数$\mathbf{W}$。每一种 $edge$ 对应一个 $\mathbf{W}_{edge}$，这样它就可以处理异构图。

但是，仔细对比GNN的GGNN的状态更新公式，细心的读者可能会发现：在GNN里需要作为输入的结点特征 $\mathbf{x}_v$ 没有出现在GGNN的公式中! 但实际上，这些结点特征对我们的预测至关重要，因为它才是各个结点的根本所在。

为了处理这个问题，GGNN将结点特征作为隐藏状态初始化的一部分。那么重新回顾一下GGNN的流程，其实就是这样：
- 用结点特征初始化各个结点的(部分)隐藏状态。
- 对整张图，按照上述状态更新公式固定迭代若干步。
- 对部分有监督信号的结点求得模型损失，利用BPTT算法反向传播求得$\mathbf{W}_{edge}$和GRU参数的梯度。

### 实例1:到达判断

为了便于理解，我们举个来自论文[10]的例子。比如说给定一张图$G$，开始结点 $S$，对于任意一个结点 $E$，模型判断开始结点是否可以通过图游走至该结点。同样地，这也可以转换成一个对结点的二分类问题，即`可以到达`和`不能到达`。下图即描述了这样的过程：

![GGNN实例](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-7-ggnn.png)

图中的红色结点即开始结点$S$，绿色结点是我们希望判断的结点$E$，我们这里称其为结束结点。那么相比于其他结点，这两个结点具有一定特殊性。那我们就可以使用第1维为1来表示开始结点，第2维为1来表示结束结点。最后在对结束结点分类时，如果其隐藏状态的第1维被赋予得到了一个非0的实数值，那意味着它可以到达。

从初始化的流程我们也可以看出GNN与GGNN的区别：GNN依赖于不动点理论，所以每个结点的隐藏状态即使使用**随机初始化都会收敛到不动点**；GGNN则不同，不同的初始化对GGNN最终的结果影响很大。

### 实例2:语义解析

上面这个例子非常简单形象地说明了GNN与GGNN的不同，由于笔者比较关注Semantic Parsing(语义解析)相关的工作，所以下面我们借用ACL 2019的一篇论文[11]来讲一下GGNN在实际中如何使用，以及它适用于怎样的场景。

首先为不了解语义解析的读者科普一下，语义解析的主要任务是将自然语言转换成机器语言，在这里笔者特指的是SQL(结构化查询语言，Structured Query Language)，它就是大家所熟知的数据库查询语言。这个任务有什么用呢？它可以让小白用户也能从数据库中获得自己关心的数据。正是因为有了语义解析，用户不再需要学习SQL语言的语法，也不需要有编程基础，可以直接通过自然语言来查询数据库。事实上，语义解析放到今天仍然是一个非常难的任务。除去自然语言与程序语言在语义表达上的差距外，很大一部分性能上的损失是因为任务本身，或者叫SQL语言的语法太复杂。比如我们有两张表格，一张是学生的学号与其性别，另一张表格记录了每个学生选修的课程。那如果想知道有多少女生选修了某门课程，我们需要先将两张表格联合(JOIN)，再对结果进行过滤(WHERE)，最后进行聚合统计(COUNT)。这个问题在多表的场景中尤为突出，每张表格互相之间通过外键相互关联。其实呢，如果我们把表格中的Header看作各个结点，表格内的结点之间存在联系，而外键可以视作一种特殊的边，这样就可以构成一张图，正如下图中部所示：

![GGNN语义解析实例](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-8-ggnn-example2.png)

论文[11]就是利用了表格这样的特性，利用GGNN来解决多表问题。下面我们先介绍一下一般的语义解析方法，再介绍[11]是如何将图跟语义解析系统联系在一起的。就笔者知道的而言，目前绝大部分语义解析会遵循Seq2seq(序列到序列，Sequence to sequence)的框架，输入是一个个自然语言单词，输出是一个个SQL单词。但这样的框架完全没有考虑到表格对SQL输出暗含的约束。比如说，在单个SELECT子句中，我们选择的若干Header都要来自同一张表。再举个例子，能够JOIN的两张表一定存在外键的联系，就像我们刚刚举的那个学生选课的例子一样。

那么，GGNN该如何结合到传统的语义解析方法中去呢？在论文[11]中，是通过三步来完成的：
1. 首先，通过表格建立对应的Graph。再利用GGNN的方法计算每个Header的隐藏状态。
2. 然后，在Seq2seq模型的编码阶段(Encoding)，用每个输入的自然语言单词的词向量对表格所有Header的隐藏状态算一个Attention，利用Attention作为权重得到了每个自然语言单词的图感知的表示。
3. 在解码阶段(Decoding)，如果输出的是表格中的Header/Table这类词，就用输出的向量与表格所有Header/Table的隐藏状态算一个分数，这个分数由$F$提供的。$F$实际上是一个全连接层，它的输出实际上是SQL单词与表格中各个Header/Table的联系程度。为了让SQL的每个输出都与历史的信息一致，每次输出时都用之前输出的Header/Table对候选集中的Header/Table打分，这样就利用到了多表的信息。

最终该论文在多表上的效果也确实很好，下面放一个在Spider[12]数据集上的性能对比：

|Model|Acc|Single|Multi|
|---|---|---|---|
|No GNN|34.9%|52.3%|14.6%|
|GNN|**40.7%**|52.2%|**26.8%**|

### GNN与GGNN

GGNN目前得到了广泛的应用，相比于GNN，其最大的区别在于不再以不动点理论为基础，虽然这意味着不再需要迭代收敛，但同时它也意味着GGNN的初始化很重要。从笔者阅读过的文献来看，GNN后的大部分工作都转向了将GNN向传统的RNN/CNN靠拢，可能的一大好处是这样可以不断吸收来自这两个研究领域的改进。但基于原始GNN的基于不动点理论的工作非常少，至少在笔者看文献综述的时候并未发现很相关的工作。

但从另一个角度来看，虽然GNN与GGNN的理论不同，但从设计哲学上来看，它们都与循环神经网络的设计类似。
- 循环神经网络的好处在于能够处理任意长的序列，但它的计算必须是串行计算若干个时间步，时间开销不可忽略。所以，上面两种基于循环的图神经网络在更新隐藏状态时不太高效。如果借鉴深度学习中堆叠多层的成功经验，我们有足够的理由相信，**多层图神经网络**能达到同样的效果。
- 基于循环的图神经网络每次迭代时都共享同样的参数，而多层神经网络每一层的参数不同，可以看成是一个**层次化特征抽取**(Hierarchical Feature Extraction)的方法。

而在[下一篇博客](https://www.cnblogs.com/SivilTaram/p/graph_neural_network_2.html)中，我们将介绍图卷积神经网络。它摆脱了基于循环的方法，开始走向**多层图神经网络**。在多层神经网络中，**卷积神经网络**(比如152层的ResNet)的大获成功又验证了其在堆叠多层上训练的有效性，所以近几年图卷积神经网络成为研究热点。

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

[11]. Representing Schema Structure with Graph Neural Networks for Text-to-SQL Parsing, https://arxiv.org/abs/1905.06241

[12]. Spider1.0 Yale Semantic Parsing and Text-to-SQL Challenge, https://yale-lily.github.io/spider

[13]. https://www.wikiwand.com/en/Laplacian_matrix

---

在[上一篇博客](https://www.cnblogs.com/SivilTaram/p/graph_neural_network.html)中，我们简单介绍了基于循环图神经网络的两种重要模型，在本篇中，我们将着大量笔墨介绍**图卷积神经网络中的卷积操作**。接下来，我们将首先介绍一下图卷积神经网络的大概框架，借此说明它与基于循环的图神经网络的区别。接着，我们将从头开始为读者介绍卷积的基本概念，以及其在物理模型中的涵义。最后，我们将详细地介绍两种不同的卷积操作，分别为**空域卷积**和**时域卷积**，与其对应的经典模型。读者不需有任何信号处理方面的基础，傅里叶变换等概念都会在本文中详细介绍。

## 图卷积缘起

在开始正式介绍图卷积之前，我们先花一点篇幅探讨一个问题：**为什么研究者们要设计图卷积操作，传统的卷积不能直接用在图上吗？** 要理解这个问题，我们首先要理解能够应用传统卷积的**图像(欧式空间)**与**图(非欧空间)**的区别。如果把图像中的每个像素点视作一个结点，如下图左侧所示，一张图片就可以看作一个非常稠密的图；下图右侧则是一个普通的图。阴影部分代表**卷积核**，左侧是一个传统的卷积核，右侧则是一个图卷积核。卷积代表的含义我们会在后文详细叙述，这里读者可以将其理解为在局部范围内的特征抽取方法。

![GGNN语义解析实例](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-9-cnn-and-gcn.png)

仔细观察两个图的结构，我们可以发现它们之间有2点非常不一样：
- 在图像为代表的欧式空间中，结点的邻居数量都是固定的。比如说绿色结点的邻居始终是8个(边缘上的点可以做Padding填充)。但在图这种非欧空间中，结点有多少邻居并不固定。目前绿色结点的邻居结点有2个，但其他结点也会有5个邻居的情况。
- 欧式空间中的卷积操作实际上是用**固定大小可学习的卷积核**来抽取像素的特征，比如这里就是抽取绿色结点对应像素及其相邻像素点的特征。但是因为图里的邻居结点不固定，所以传统的卷积核不能直接用于抽取图上结点的特征。

真正的难点聚焦于**邻居结点数量不固定**上。那么，研究者如何解决这个问题呢？其实说来也很简单，目前主流的研究从2条路来解决这件事：
- 提出一种方式把非欧空间的图转换成欧式空间。
- 找出一种可处理变长邻居结点的卷积核在图上抽取特征。

这两条实际上也是后续图卷积神经网络的设计原则，**图卷积**的本质是想找到**适用于图的可学习卷积核**。

## 图卷积框架(Framework)

上面说了图卷积的核心特征，下面我们先来一窥图卷积神经网络的全貌。如下图所示，输入的是整张图，在`Convolution Layer 1`里，对每个结点的邻居都进行一次卷积操作，并用卷积的结果更新该结点；然后经过激活函数如`ReLU`，然后再过一层卷积层`Convolution Layer 2`与一词激活函数；反复上述过程，直到层数达到预期深度。与GNN类似，图卷积神经网络也有一个局部输出函数，用于将结点的状态(包括隐藏状态与结点特征)转换成任务相关的标签，比如水军账号分类，本文中笔者称这种任务为`Node-Level`的任务；也有一些任务是对整张图进行分类的，比如化合物分类，本文中笔者称这种任务为`Graph-Level`的任务。**卷积操作关心每个结点的隐藏状态如何更新**，而对于`Graph-Level`的任务，它们会在卷积层后加入更多操作。本篇博客主要关心如何在图上做卷积，至于如何从结点信息得到整张图的表示，我们将在下一篇系列博客中讲述。

![图卷积神经网络全貌](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-10-gcn-framework.png)

多说一句，GCN与GNN乍看好像还挺像的。为了不让读者误解，在这里我们澄清一下它们根本上的不同：GCN是多层堆叠，比如上图中的`Layer 1`和`Layer 2`的参数是不同的；GNN是迭代求解，可以看作每一层Layer参数是共享的。

## 卷积(Convolution)

正如我们在上篇博客的开头说到的，图卷积神经网络主要有两类，一类是基于**空域**的，另一类则是基于**频域**的。通俗点解释，空域可以类比到直接在图片的像素点上进行卷积，而频域可以类比到对图片进行傅里叶变换后，再进行卷积。傅里叶变换的概念我们先按下不讲，我们先对两类方法的代表模型做个大概介绍。

基于空域卷积的方法直接将卷积操作定义在每个结点的连接关系上，它跟传统的卷积神经网络中的卷积更相似一些。在这个类别中比较有代表性的方法有 Message Passing Neural Networks(MPNN)[1], GraphSage[2], Diffusion Convolution Neural Networks(DCNN)[3], PATCHY-SAN[4]等。

基于频域卷积的方法则从图信号处理起家，包括 Spectral CNN[5], Cheybyshev Spectral CNN(ChebNet)[6], 和 First order of ChebNet(1stChebNet)[7]等。

在介绍这些具体的模型前，先让我们从不同的角度来回顾一下卷积的概念，重新思考一下卷积的本质。

### 基础概念

由维基百科的介绍我们可以得知，**卷积**是一种定义在两个函数($f$跟$g$)上的数学操作，旨在产生一个新的函数。那么$f$和$g$的卷积就可以写成$f*g$，数学定义如下：

$$(f*g)(t)={\int}_{-\infty}^{\infty}f(\tau)g(t-\tau) (连续形式)$$
$$(f*g)(t)={\sum}_{\tau=-\infty}^{\infty}f(\tau)g(t-\tau) (离散形式)$$

### 实例:掷骰子问题

光看数学定义可能会觉得非常抽象，下面我们举一个掷骰子的问题，该实例参考了知乎问题"如何通俗易懂地解释卷积"[8]的回答。

想象我们现在有两个骰子，两个骰子分别是$f$跟$g$，$f(1)$表示骰子$f$向上一面为数字$1$的概率。同时抛掷这两个骰子1次，它们正面朝上数字和为4的概率是多少呢？相信读者很快就能想出它包含了三种情况，分别是：
- $f$ 向上为1，$g$ 向上为3；
- $f$ 向上为2，$g$ 向上为2；
- $f$ 向上为3，$g$ 向上为1；

最后这三种情况出现的概率和即问题的答案，如果写成公式，就是 $\sum_{\tau=1}^{3}f(\tau)g(4-\tau)$。可以形象地绘制成下图：

![卷积基本概念](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-11-convolution-basic.png)

如果稍微扩展一点，比如说我们认为 $f(0)$ 或者 $g(0)$ 等是可以取到的，只是它们的值为0而已。那么该公式可以写成$\sum_{\tau=-\infty}^{\infty}f(\tau)g(4-\tau)$。仔细观察，这其实就是卷积$(f*g)(4)$。如果将它写成内积的形式，卷积其实就是 $[f(-\infty),\cdots,f(1),\cdots,f(\infty)] \cdot [g(\infty),\cdots,g(3),\cdots,g(-\infty)]$。这么一看，是不是就对卷积的名字理解更深刻了呢? 所谓卷积，其实就是把一个函数卷(翻)过来，然后与另一个函数求内积。

对应到不同方面，卷积可以有不同的解释：$g$ 既可以看作我们在深度学习里常说的**核**(Kernel)，也可以对应到信号处理中的**滤波器**(Filter)。而 $f$ 可以是我们所说的机器学习中的**特征**(Feature)，也可以是信号处理中的**信号**(Signal)。f和g的卷积 $(f*g)$就可以看作是对$f$的加权求和。下面两个动图就分别对应信号处理与深度学习中卷积操作的过程[9][10]。

![信号处理中的卷积](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-12-conv-signal.gif)

![深度学习中的卷积](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-13-conv-cnn.gif)

## 空域卷积(Spatial Convolution)

介绍完卷积的基础概念后，我们先来介绍下**空域卷积**(Spatial Convolution)。从设计理念上看，空域卷积与深度学习中的卷积的应用方式类似，其核心在于**聚合邻居结点的信息**。比如说，一种最简单的无参卷积方式可以是：将所有直连邻居结点的隐藏状态加和，来更新当前结点的隐藏状态。

![最简单的空域卷积](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-12-basic-spatial-conv.png)

> 这里非参式的卷积只是为了举一个简单易懂的例子，实际上图卷积在建模时需要的都是带参数、可学习的卷积核。

### 消息传递网络(Message Passing Neural Network)

消息传递网络(MPNN)[1] 是由Google科学家提出的一种模型。严格意义上讲，MPNN不是一种具体的模型，而是一种空域卷积的形式化框架。它将空域卷积分解为两个过程：**消息传递**与**状态更新**操作，分别由$M_{l}(\cdot)$和$U_{l}(\cdot)$函数完成。将结点$v$的特征$\mathbf{x}_v$作为其隐藏状态的初始态$\mathbf{h}_{v}^0$后，空域卷积对隐藏状态的更新由如下公式表示：

$$\mathbf{h}_{v}^{l+1}=U_{l+1}(\mathbf{h}_v,\sum_{u{\in}ne[v]}M_{l+1}(\mathbf{h}_v^l,\mathbf{h}_u^l,\mathbf{x}_{vu}))$$

其中$l$代表图卷积的第$l$层，上式的物理意义是：收到来自每个邻居的的消息$M_{l+1}$后，每个结点如何更新自己的状态。

如果读者还记得GGNN的话，可能会觉得这个公式与GGNN的公式很像。实际上，它们是截然不同的两种方式：GCN中通过级联的层捕捉邻居的消息，GNN通过级联的时间来捕捉邻居的消息；前者层与层之间的参数不同，后者可以视作层与层之间共享参数。MPNN的示意图如下[11]：

![MPNN网络模型](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-13-mpnn.png)

### 图采样与聚合(Graph Sample and Aggregate)

MPNN很好地概括了空域卷积的过程，但定义在这个框架下的所有模型都有一个共同的缺陷：卷积操作针对的对象是**整张图**，也就意味着要将所有结点放入内存/显存中，才能进行卷积操作。但对实际场景中的大规模图而言，整个图上的卷积操作并不现实。GraphSage[2]提出的动机之一就是解决这个问题。从该方法的名字我们也能看出，区别于传统的全图卷积，GraphSage利用**采样**(Sample)部分结点的方式进行学习。当然，即使不需要整张图同时卷积，GraphSage仍然需要聚合邻居结点的信息，即论文中定义的*aggregate*的操作。这种操作类似于MPNN中的**消息传递**过程。

![GraphSage采样过程](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-14-graphsage.jpg)

具体地，GraphSage中的采样过程分为三步:
1. 在图中随机采样若干个结点，结点数为传统任务中的`batch_size`。对于每个结点，随机选择固定数目的邻居结点(这里邻居不一定是一阶邻居，也可以是二阶邻居)构成进行卷积操作的图。
2. 将邻居结点的信息通过$aggregate$函数聚合起来更新刚才采样的结点。
3. 计算采样结点处的损失。如果是无监督任务，我们希望图上邻居结点的编码相似；如果是监督任务，即可根据具体结点的任务标签计算损失。

最终，GraphSage的状态更新公式如下：

$$\mathbf{h}_{v}^{l+1}=\sigma(\mathbf{W}^{l+1}\cdot aggregate(\mathbf{h}_v^l,\{\mathbf{h}_u^l\}),{\forall}u{\in}ne[v])$$

GraphSage的设计重点就放在了$aggregate$函数的设计上。它可以是不带参数的$max$, $mean$, 也可以是带参数的如$LSTM$等神经网络。核心的原则仍然是，它需要可以处理变长的数据。在本系列博客的第三篇笔者会介绍卷积神经网络中针对图任务的`ReadOut`操作，$aggregate$函数的设计与其有异曲同工之妙，此处就不展开叙述了。

### 图结构序列化(PATCHY-SAN)

我们之前曾提到卷积神经网络不能应用在图结构上是因为图是非欧式空间，所以大部分算法都沿着**找到适用于图**的卷积核这个思路来走。而 PATCHY-SAN 算法 [4] 另辟蹊径，它将图结构转换成了序列结构，然后直接利用卷积神经网络在转化成的序列结构上做卷积。由于 PATCHY-SAN在其论文中主要用于图的分类任务，我们下面的计算过程也主要针对图分类问题(例如，判断某个社群的职业)。

那么，图结构转换成序列结构最主要的挑战在何处呢，如果简单的话，为什么以前的工作没有尝试把图转成序列结构呢？就笔者个人的观点来看，这种序列转换要保持图结构的两个特点：1. 同构的性质。 2. 邻居结点的连接关系。对于前者而言，意味着当我们把图上结点的标号随机打乱，得到的仍应是一样的序列。简单来说就是，同构图产生的序列应当相似，甚至一样；对于后者，则意味着我们要保持邻居结点与目标结点的距离关系，如在图中的三阶邻居在序列中不应该成为一阶邻居等。

PATCHY-SAN 通过以下三个步骤来解决这两个问题：

1. **结点选择(Node Squenece Selection)**。该过程旨在与通过一些人为定义的规则(如度大的结点分数很高，邻居的度大时分数较高等)为每个结点指定一个在图中的排序。在完成排序后，取出前 $\omega$ 个结点作为整个图的代表。
2. **邻居结点构造(Neighborhood graph construction)**。在完成结点排序后，以第1步选择的结点为中心，得到它们的邻居(这里的邻居可以是第一阶邻居，也可以是二阶邻居)结点，就构成了 $\omega$ 个团。根据第1步得到的结点排序对每个团中的邻居结点进行排序，再取前 $k$ 个邻居结点按照顺序排列，即组成 $\omega$ 个有序的团。
3. **图规范化(Graph Noermalization)**。按照每个团中的结点顺序可将所有团转换成固定长度的序列($k+1$)，再将它们按照中心结点的排序从前到后依次拼接，即可得到一个长度为 ${\omega}*(k+1)$ 的代表整张图的序列。这样，我们就可以直接使用带1D的卷积神经网络对该序列建模，比如图分类(可类比文本序列分类)。值得注意的一点是，在第1步和第2步中，如果取不到 $\omega$ 或 $k$ 个结点时，要使用空结点作填充(padding)。

一个形象的流程图如下所示，图源自论文[4]。

![Pathcy-san framework](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-14-pathcy-san-framework.png)

下图可能可以帮助读者更好地理解这种算法，图来自[12]。整个流程自底向上：首先根据自定义规则对图里的结点进行排序，然后选择前6个结点，即图中的 1至6；接着我们把这些结点

![Patchy-san 具体样例](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-15-pathcy-san-detail.png)

## 频域卷积(Spectral Convolution)

空域卷积非常直观地借鉴了图像里的卷积操作，但据笔者的了解，它缺乏一定的理论基础。而频域卷积则不同，相比于空域卷积而言，它主要利用的是**图傅里叶变换(Graph Fourier Transform)**实现卷积。简单来讲，它利用图的**拉普拉斯矩阵(Laplacian matrix)**导出其频域上的的拉普拉斯算子，再类比频域上的欧式空间中的卷积，导出图卷积的公式。虽然公式的形式与空域卷积非常相似，但频域卷积的推导过程却有些艰深晦涩。接下来我们将攻克这部分看起来很难的数学公式，主要涉及到**傅里叶变换(Fourier Transform)**和**拉普拉斯算子(Laplacian operator)**。即使读者没有学过任何相关知识也不要紧，笔者将尽可能用形象的描述解释每个公式的涵义，让读者能感悟这些公式的美妙之处。

### 前置内容

如上所述，在本小节，我们将介绍两个主要的知识点：傅里叶变换与拉普拉斯算子。在介绍之前，我们先抛出两个问题：1. 什么是傅里叶变换; 2. 如何将傅里叶变换扩展到图结构上。这两个问题是前置内容部分要解决的核心问题，读者可带着这两个问题，完成下面内容的阅读。

#### 傅里叶变换(Fourier Transform)

借用维基百科的说法，**傅里叶变换(Fourier Transform, FT)**会将一个在空域(或时域)上定义的函数分解成频域上的若干频率成分。换句话说，傅里叶变换可以将一个函数从空域变到频域。先抛开傅里叶变换的数学公式不谈，用 $F$ 来表示傅里叶变换的话，我们先讲一个很重要的恒等式：

$$(f*g)(t)=F^{-1}[F[f(t)]{\odot}F[g(t)]]$$

这里的$F^{-1}$指的是傅里叶逆变换，$\odot$是哈达玛乘积，指的是两个矩阵(或向量)的**逐点乘积(Element-wise Multiplication)**。仔细观察上面这个公式，它的直观含义可以用一句话来概括：*空(时)域卷积等于频域乘积*。简单来说就是，如果要算 $f$ 与 $g$ 的卷积，可以先将它们通过傅里叶变换变换到频域中，将两个函数在频域中相乘，然后再通过傅里叶逆变换转换出来，就可以得到 $f$ 与 $g$ 的卷积结果。下面的动图形象地展示了傅里叶变换的过程，这里我们把函数 $f$ 傅里叶变换后的结果写作 $\hat{f}$.

![傅里叶变换的示例](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-15-ft-example.gif)

那傅里叶变换能干啥呢，有一个简单的应用是给图像去除一些规律噪点。比如说下面这个例子，原图来自知乎 [13]。

在傅里叶变换前，图像上有一些规律的条纹，直接在原图上去掉条纹有点困难，但我们可以将图片通过傅里叶变换变到频谱图中，频谱图中那些规律的点就是原图中的背景条纹。

![傅里叶变换的示例](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-16-ft-transoform-before.jpg)

只要在频谱图中擦除这些点，就可以将背景条纹去掉，得到下图右侧的结果。

![傅里叶变换的示例](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-17-ft-transoform-after.jpg)

除了可以用来分离噪声点与正常点，傅里叶变换还凭借上面的恒等式，在加速卷积运算方面有很大的潜力，**快速傅里叶变换(Fast Fourier Transform)**也是由此而生。实际上呢，现在大家最常用的卷积神经网络，完全可以搭配傅里叶变换。下面这张图就表示了一个普通的卷积神经网络如何与傅里叶变换搭配，其中的 IFFT 即 **快速傅里叶变换的逆变换(Inverse Fast Fourier Transform**：

![傅里叶变换的示例](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-18-ft-cnn-example.png)

其实笔者在初识傅里叶变换时很好奇，既然FFT可以加速卷积神经网络，为什么现在的卷积神经网络不用呢? 在经过一些搜索与思考后，笔者将自己得到的结论抛砖引玉供读者参考：我们现在的卷积神经网络的核都很小，常见的都如1，3，5之类，卷积操作的时间开销本来就不大。如果要搭配FFT，还需要做傅里叶变换与逆变换，时间开销并不一定会减小。

说了这么半天，傅里叶变换的公式是什么样呢？实际上，$f$经过傅里叶变换后的结果$\hat{f}$就如下所示，其中 $i=\sqrt{-1}$(虚数单位)，t是任意实数。

$$\hat{f}(t)={\int}f(x){\exp}^{-2{\pi}ixt}dx$$

感兴趣的同学可以深入研究一下傅里叶变换这一套，我们这里关心的实际上是 ${\exp}^{-2{\pi}ixt}$的物理意义，它是图上类比构造傅里叶变换的关键。这个式子实际上是拉普拉斯算子$∆$的广义特征函数。

> 拉普拉斯算子(Laplacian operator) 的物理意义是空间二阶导，准确定义是：标量梯度场中的散度，一般可用于描述物理量的流入流出。比如说在二维空间中的温度传播规律，一般可以用拉普拉斯算子来描述。

为什么是特征函数呢，我们这里根据拉普拉斯算子的定义来稍微推导一下。众所周知，特征向量需要满足的定义式是：对于矩阵$A$，其特征向量满足的条件应是矩阵与特征向量$x$做乘法的结果，与特征向量乘标量$\lambda$的结果一样，即满足如下等式。

$$Ax={\lambda}x$$

稍微推导一下即可知道，拉普拉斯算子作用在${\exp}^{-2{\pi}ixt}$确实满足以上特征向量的定义:

$$∆{\exp}^{-2{\pi}ixt}=\frac{{\partial}^2}{{\partial}t^2}{\exp}^{-2{\pi}ixt}={-4{\pi}^2x^2}{\exp}^{-2{\pi}ixt}$$

这里$\partial$是求导符号，$\partial^2$是二阶导。

实际上，再仔细观察傅里叶变换的式子，它本质上是将函数$f(t)$映射到了以$\{{\exp}^{-2{\pi}ixt}\}$为基向量的空间中。

#### 图上的傅里叶变换

终于讲到我们本节的重点内容了，上面我们絮絮叨叨地铺垫了很多傅里叶变换的知识，主要是为了将傅里叶变换类比到图上。那么问题来了:在图上，我们去哪找拉普拉斯算子 $∆$ 与 ${\exp}^{-2{\pi}ixt}$呢？

聪明的研究者们找到了图的**拉普拉斯矩阵(L)**及其**特征向量($u$)**，作为上述两者的替代品。至此，形成了图上傅里叶变换的生态系统。拉普拉斯矩阵，实际上是**度矩阵(D)**减去**邻接矩阵(A)** $L=D-A$，如下图所示，图源自[14].

![拉普拉斯矩阵](https://raw.githubusercontent.com/SivilTaram/Graph-Neural-Network-Note/master/images/image-19-graph-laplacian-matrix)

频域卷积的前提条件是图必须是无向图，那么$L$就是对称矩阵，所以它可以按照如下公式分解:

$$L = U{\Lambda}U^{T}$$
$$U = (u_1, u_2,{\cdots},u_n)$$
$$
 {\Lambda}=\left[
 \begin{matrix}
   {\lambda}_1 & ... & 0 \\
   ... & ... & ... \\
   0 & ... & {\lambda}_n
  \end{matrix}
  \right]
$$

那么，根据上面卷积与傅里叶结合的变换公式，图上频域卷积的公式便可以写成 $\hat{f}(t)={\sum}_{n=1}^{N}f(n)u_t(n)$。如果在整个图的$N$个结点上一起做卷积，就可以得到整张图上的卷积如下：

$$
 {\hat{f}}=\left[
 \begin{matrix}
   {\hat{f}}(1) \\
   ... \\
   {\hat{f}}(N)  \end{matrix}
  \right]=U^Tf
$$

让我们重新审视一下欧式空间上的卷积和图上的卷积，即可明白图上的卷积与传统的卷积其实非常相似，这里 $f$ 都是特征函数，$g$ 都是卷积核：

$$(f*g)=F^{-1}[F[f]{\odot}F[g]]$$
$$(f{*}_{G}g)=U(U^Tf{\odot}U^Tg)=U(U^Tg{\odot}U^Tf)$$

如果把 $U^Tg$ 整体看作可学习的卷积核，这里我们把它写作 $g_{\theta}$。最终的卷积公式即：

$$
o = (f*_{G}g)_{\theta}
$$

## 参考文献

[1]. Neural Message Passing for Quantum Chemistry, https://arxiv.org/abs/1704.01212

[2]. Inductive Representation Learning on Large Graphs, https://arxiv.org/abs/1706.02216

[3]. Diffusion-Convolutional Neural Networks, https://arxiv.org/abs/1511.02136

[4]. Learning Convolutional Neural Networks for Graphs, https://arxiv.org/pdf/1605.05273

[5]. Spectral Networks and Locally Connected Networks on Graphs, https://arxiv.org/abs/1312.6203

[6]. Convolutional neural networks on graphs with fast localized spectral filtering, https://papers.nips.cc/paper/6081-convolutional-neural-networks-on-graphs-with-fast-localized-spectral-filtering

[7]. Semi-Supervised Classification with Graph Convolutional Networks, https://arxiv.org/pdf/1609.02907

[8]. 如何通俗易懂地解释卷积, https://www.zhihu.com/question/22298352

[9]. https://en.wikipedia.org/wiki/Convolution

[10]. https://mlnotebook.github.io/post/CNN1/

[11]. http://snap.stanford.edu/proj/embeddings-www

[12]. https://zhuanlan.zhihu.com/p/37840709

[13]. https://www.zhihu.com/question/20460630/answer/105888045

[14]. https://en.wikipedia.org/wiki/Laplacian_matrix

---

在上篇博客中我们仔细介绍了
