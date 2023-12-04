# 虚假新闻检测实验报告

### 2011181 &nbsp; &nbsp; 唐鹏程

## 一. 问题分析
虚假新闻危害着人们的生活，但是人工对虚假新闻进行检测会消耗大量的人力物力和时间，因此自动的虚假新闻检测是十分有必要的。

虚假新闻检测是一个复杂的任务，面临着多方面的挑战：

+ 首先，定义虚假新闻本身就是一个具有挑战性的问题。虚假新闻包括但不限于夸张报道、虚构信息、误导性标题等。这种多样性使得确定一个明确的定义变得困难，因为它可能在不同的语境中有不同的解释。

+ 其次，我们使用的数据集可能存在一些局限性。数据集的规模可能不足以涵盖虚假新闻的所有方面，而且虚假新闻的创作者可能会采用不同的策略来规避检测。

+ 此外，数据集中虚假新闻和真实新闻的分布可能不平衡。复杂性也体现在语言理解的挑战上。虚假新闻的作者通常使用一些语言上的技巧，例如双关语、歧义性的表达和隐喻，这对于计算机模型来说可能是具有挑战性的。

本实验作为对虚假新闻检测的一个入门性学习，只通过机器学习实现了一个较为简易的虚假新闻检测器，采用了几种不同的模型分别测试，对比模型的效果。在评估模型时，我选择了精确度，AUC和召回率等指标。然而，这些指标并不能完全捕捉到虚假新闻检测任务的所有方面，特别是当虚假新闻和真实新闻的样本分布不平衡时。


## 二. 数据集说明
本实验采用的数据集是中文微信消息，包括微信消息的`Official Account Name`，`Title，News Url`，`Image Url`，`Report Content`，`label`。`Title`是微信消息的标题，`label`是消息的真假标签（0是real消息，1是fake消息）。训练数据保存在train.news.csv，测试数据保存在test.feature.csv。

数据来源：[Wang, Y., Yang, W., Ma, F., Xu, J., Zhong, B., Deng, Q., & Gao, J. (2020). Weak Supervision for Fake News Detection via Reinforcement Learning. Proceedings of the AAAI Conference on Artificial Intelligence, 34(01), 516-523](https://pure.psu.edu/en/publications/weak-supervision-for-fake-news-detection-via-reinforcement-learni).

## 三. 方法介绍
本实验主要步骤分为数据处理、特征构建和模型训练，在方法的选择上，特征构建我选择了TF-IDF特征构建，
模型的选择方面我尝试了朴素贝叶斯和岭回归、逻辑回归等方法

### TF-IDF特征构建
TF-IDF 即词频-逆文档频率（Term Frequency-Inverse Document Frequency），是一种用于信息检索和文本挖掘的常用加权技术。我们先定义词频和逆文档频率两个概念：

* __词频__：$TF(t,d)=\frac{词t在文档d中出现的次数}{包含词t的文档数+1}$

* __逆文档频率__：$IDF(t,D)=\log{(\frac{文档集合D的总文档数}{包含词t的文档数+1})}$

* __TF-IDF__：$TF-IDF(t,d,D)=TF(t,d)\cdot IDF(t,D)$

TF-IDF 的目标是将那些在特定文档中频繁出现但在整个文档集中很少出现的词赋予更高的权重。

### 朴素贝叶斯
朴素贝叶斯（Naive Bayes）是一种基于贝叶斯定理的分类算法。它被称为“朴素”是因为它假设特征之间相互独立，这是一个较强的假设，因此称为“朴素”。在朴素贝叶斯分类中，先计算每个类别的先验概率，即在不考虑任何特征的情况下某一类别出现的概率。再计算每个特征在给定类别下的条件概率$P(F_j|C_i)$，即在已知样本属于类别 $C_i$ 的情况下特征 $F_j$ 出现的概率。给定一个新的样本，使用贝叶斯定理计算它属于每个类别的概率，并选择概率最大的类别作为预测结果，公式如下：
$$ P(C_i|样本特征) \propto P(C_i)\prod_{j=1}^n{P(F_j|C_i)} $$

#### 1. 多项式朴素贝叶斯
多项式朴素贝叶斯假设特征的多项分布，这意味着每个特征的取值是一个离散的非负整数，通常表示特征的出现次数或频率。在文本分类中，典型的特征是单词，而每个特征的取值是单词在文档中的出现次数或频率。

多项式朴素贝叶斯和标准贝叶斯的区别主要在于条件概率的计算，在多项式模型中，通常使用拉普拉斯平滑（Laplace smoothing）来处理概率为零的情况：

$$ P(F_j|C_i) = \frac{属于类别 C_i 且具有特征 F_j 的样本数+1}{属于类别 C_i 的样本数+特征总数} $$

多项式朴素贝叶斯适用于文本分类等问题，处理离散的特征。但缺点在于对于大规模和高维度的数据，可能出现过拟合，且对特征之间的依赖关系做了较强的假设。

#### 2. 伯努利朴素贝叶斯
伯努利朴素贝叶斯是朴素贝叶斯分类器的另一种变体，它主要用于处理二元数据，即特征只有两种取值。与多项式朴素贝叶斯不同，伯努利朴素贝叶斯关注的是特征是否存在，而不是特征的频率。

在伯努利模型中，每个特征的取值只能是0或1，表示特征的缺失或存在。在文本分类中，特征通常是单词，取值为1表示文档中包含该单词，取值为0表示文档中不包含该单词。

条件概率计算公式为：
$$ P(F_j|C_i) = \frac{属于类别 C_i 且具有特征 F_j 的样本数+1}{属于类别 C_i 的样本数+2} $$


### 岭回归
岭回归（Ridge Regression）是一种用于处理多重共线性问题的线性回归扩展方法。多重共线性指的是自变量之间存在高度相关性的情况，这会导致普通线性回归模型的系数估计变得不稳定，对数据中的噪声敏感。

岭回归通过在普通线性回归的损失函数中添加一个正则化项，以限制模型系数的增长，从而缓解多重共线性问题。

损失函数如下：
$$ Loss = \sum_{i=1}^{N}(y_i-\hat{y_i})^2+\alpha \sum_{j=1}^{p}\beta_j^2 $$

其中，$\alpha$是岭回归的超参数，用于调节正则化的强度，$\beta_j$是模型的系数。

岭回归优点是能缓解多重共线性问题，提高模型的稳定性，
且对数据集中的噪声不敏感。但也有缺点，比如需要事先设定正则化参数 $\alpha$，且不适用于特征数量远大于样本数量的情况。

### 逻辑回归
逻辑回归（Logistic Regression）实际上是一种用于二分类问题的统计学习方法，尽管其名称中包含"回归"一词，但它实际上是一种分类算法而不是回归算法。其主要思想是利用 logistic 函数将线性组合的特征映射到[0, 1]的概率范围。这个概率表示某个样本属于正类别的可能性。

Logistic函数：
$$ \sigma(z)=\frac{1}{1+e^{-z}} $$
其中 $z$ 是线性组合： $$ z = \beta_0+\beta_1x_1+\beta_2x_2+...+\beta_nx_n$$

逻辑回归模型可以表示为：
$$ P(Y=1|X)=\frac{1}{1+e^{-(\beta_0+\beta_1x_1+\beta_2x_2+...+\beta_nx_n)}} $$

损失函数采用对数似然损失函数，其目标是最大化观测到的样本的对数似然：
$$ Loss=-\sum_{i=1}^{N}[y_i\log(\hat p_i)+(1-y_i)\log(1-\hat p_i)] $$

其中，$y_i$是样本的实际类别（0 或 1），$\hat p_i$是样本属于正类别的预测概率。

逻辑回归的参数学习通常使用梯度下降等优化算法，目标是最小化损失函数。

逻辑回归简单而有效，容易理解和实现且适用于二分类问题。但对于特征之间存在高度相关性的情况，逻辑回归可能不表现得很好。

## 四. 关键代码细节
* 首先是数据处理，读取训练数据后选取 ‘Title’，’Official Account Name‘，‘Report Content’三列进行合并，并划分训练集和测试集
    ```python
    data['Text'] = data['Title'].astype(str) + data['Ofiicial Account Name'].astype(str) + data['Report Content'].astype(str)

    # 分割训练集和测试集
    x_train, x_test,y_train,y_test = train_test_split(data['Text'], labels, test_size=0.3, random_state=20)
    ```

* 然后对 x_test 和 x_train 分别进行分词处理，方便后续构建特征矩阵，具体方法选用的是 jieba ，一个很强大的 Python 中文分词库，jieba.cut()方法给定中文字符串，分解后返回一个迭代器
    ```python
    # 分词
    for i in x_train.keys():
        sentences = x_train[i].split()
        sent_words = [list(jieba.cut(sent0)) for sent0 in sentences]
        words = ""
        for sent0 in sent_words:
            words = " ".join(sent0) 
        x_train[i] = words
    ```

* 调用 TfidfVectorizer 库函数进行 TF-IDF 特征构建，其中fit()方法加载数据，并计算tf-idf值，而transform()方法将数据转化为matrix的形式，对于需要转化为矩阵的训练集，可以直接调用fit_transform()方法将两步合并
    ```python
    # 文本特征提取，建立特征权重
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', max_df=0.5, norm=None)

    # 建立数据特征矩阵
    tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
    tfidf_test = tfidf_vectorizer.transform(x_test)
    ```

* 对数据集的处理已经准备完毕，接下来直接调用模型进行训练即可，我先后尝试了多项式朴素贝叶斯(MultinomialNB)，伯努利朴素贝叶斯(BernoulliNB)，岭回归(RidgeClassifier)，逻辑回归(LogisticRegression)四种模型
    ```python
    # clf = LogisticRegression(random_state=32)
    # clf = RidgeClassifier(alpha=1.0)
    # clf = BernoulliNB()
    clf = MultinomialNB()
    clf.fit(tfidf_train, y_train)
    y_pred = clf.predict(tfidf_test)
    ```
* 最后输出评价指标，分别选用了精确度、召回率、F1_score和AUC值，通过调用 metrics 库，将 y_test 与通过模型预测得到的 y_pred 值比对即可
    ```python
    # 评价指标
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("F1_score:", metrics.f1_score(y_test, y_pred))
    print("AUC:", metrics.roc_auc_score(y_test, y_pred))
    ```
 
## 五. 运行截图

## 六. 实验结果分析
