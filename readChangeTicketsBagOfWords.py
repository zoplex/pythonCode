
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('display.width', 300)


ch1k = pd.read_csv("C:/Users/zkrunic/Documents/BigData/ML/DSU/DSU-ML-2018/dataCommon/SM_CHANGES.csv", nrows=10000)
ch1k.describe()
ch1k.head()
list(ch1k)
ch1k = ch1k.dropna(subset=['Outage Comments'])
corpus = ch1k['Outage Comments']
bow = CountVectorizer()
X = bow.fit_transform(corpus)
bow.get_feature_names()
X.toarray()
X.shape
bow.get_feature_names()

alltext = ' '.join(corpus)

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
nltk.download('stopwords')
stop = stopwords.words('english') + list(string.punctuation)


cleantext = [i for i in word_tokenize(alltext.lower()) if i not in stop]
freq = nltk.FreqDist(cleantext)
# Print and plot most common words
freq.most_common(20)
freq.plot(10, title="Most common words in the change tickets description field")

bigrm = nltk.bigrams(cleantext)
freq2 = nltk.FreqDist(bigrm)
for k,v in freq2.items():
    print(k,v)

freq2.most_common(20)
freq2.plot(10, title="Most common bigrams in the change tickets description field")





