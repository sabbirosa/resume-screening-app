import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('UpdatedResumeDataSet.csv')


def clean_resume(text):
    clean_text = re.sub('http\S+\s', ' ', text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+\s', ' ', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape(
        """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text


df['Resume'] = df['Resume'].apply(lambda x: clean_resume(x))

le = LabelEncoder()

le.fit(df['Category'])
df['Category'] = le.transform(df['Category'])

tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(df['Resume'])
required_text = tfidf.transform(df['Resume'])

x_train, x_test, y_train, y_test = train_test_split(
    required_text, df['Category'], test_size=0.2, random_state=42)

clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(x_train, y_train)
ypred = clf.predict(x_test)

pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
pickle.dump(clf, open('clf.pkl', 'wb'))
