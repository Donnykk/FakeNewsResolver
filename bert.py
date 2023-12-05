import pandas as pd 
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

mod = int(input())
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
    
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
a_tokens = bert_tokenizer.tokenize(x_train)
print(a_tokens)