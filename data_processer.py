import pandas as pd 
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def process_data(mod):
    data = pd.read_csv(r'./train.news.csv', encoding='utf-8')
    labels = data.label
    data['Text'] = data['Title'].astype(str) + data['Ofiicial Account Name'].astype(str) + data['Report Content'].astype(str)

    # 分割训练集和测试集
    x_train, x_test,y_train,y_test = train_test_split(data['Text'], labels, test_size=0.01, random_state=20)
    
    test_data = pd.read_csv(r'./test.feature.csv', encoding='utf-8')
    x_test = test_data['Title'].astype(str) + test_data['Ofiicial Account Name'].astype(str) + test_data['Report Content'].astype(str)
    y_test = test_data['id']
    
    if mod == 1:
        data = pd.read_csv(r'./train.news.csv', encoding='utf-8')
        labels = data.label
        data['Text'] = data['Title'].astype(str) + data['Ofiicial Account Name'].astype(str) + data['Report Content'].astype(str)
        x_train, x_test,y_train,y_test = train_test_split(data['Text'], labels, test_size=0.3, random_state=20)
    
    # 分词
    for i in x_train.keys():
        sentences = x_train[i].split()
        sent_words = [list(jieba.cut(sent0)) for sent0 in sentences]
        words = ""
        for sent0 in sent_words:
            words = " ".join(sent0) 
        x_train[i] = words
    for i in x_test.keys():
        sentences = x_test[i].split()
        sent_words = [list(jieba.cut(sent0)) for sent0 in sentences]
        words = ""
        for sent0 in sent_words:
            words = " ".join(sent0) 
        x_test[i] = words

    # 文本特征提取，建立特征权重
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', max_df=0.5, norm=None)

    # 建立数据特征矩阵
    tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
    tfidf_test = tfidf_vectorizer.transform(x_test)
    return tfidf_train, tfidf_test, y_train, y_test



