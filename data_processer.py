import pandas as pd
import zhon
import string
import nltk
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 

data = pd.read_csv("/home/donny/Desktop/News_Resolver/train.news.csv", on_bad_lines='skip')
labeled_data = data[['Title', 'label']]

 
for str in labeled_data['Title']:
    # filter
    for c in zhon.hanzi.punctuation:
        str = str.replace(c, " ")
    for c in string.punctuation:
        str = str.replace(c, " ")
    s = re.compile('[\\x00-\\x08\\x0b-\\x0c\\x0e-\\x1f]').sub(' ', str)
    str = s
    
    # divide word
    word_list = nltk.word_tokenize(str)
    filtered = [w for w in word_list if w not in stopwords.words('chinese')]
    ps = PorterStemmer()
    filtered = [ps.stem(w) for w in filtered]  
    wl = WordNetLemmatizer()   
    filtered = [wl.lemmatize(w) for w  in filtered]  
    str = " ".join(filtered)
    print(str)
    
    