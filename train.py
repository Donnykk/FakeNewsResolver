import data_processer
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics  

# 运行模式
mod = int(input())

# 数据处理
tfidf_train, tfidf_test, y_train, y_test = data_processer.process_data(mod)

# 调用模型      
# clf = LogisticRegression(random_state=32)
# clf = RidgeClassifier(alpha=1.0)
# clf = BernoulliNB()
# clf = MultinomialNB()
clf = MLPClassifier(solver='sgd', activation='logistic', alpha=1e-4, hidden_layer_sizes=(50, 50), random_state=1, max_iter=100, verbose=True, learning_rate_init=.1)
clf.fit(tfidf_train, y_train)
y_pred = clf.predict(tfidf_test)

# 保存至csv
np.savetxt('/home/donny/Desktop/FakeNewsResolver/predict_data.csv', y_pred, fmt='%i')
df = pd.read_csv('/home/donny/Desktop/FakeNewsResolver/predict_data.csv', encoding='utf-8', header=None)
df.columns = ['label']
df.insert(0, 'id', range(1, len(df)+1))
df.to_csv('/home/donny/Desktop/FakeNewsResolver/predict_data.csv', index=False)

# 评价指标
if mod == 1:
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("F1_score:", metrics.f1_score(y_test, y_pred))
    print("AUC:", metrics.roc_auc_score(y_test, y_pred))