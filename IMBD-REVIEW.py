import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

df = pd.read_csv('my1.csv')
def clean1(text):
  soup = BeautifulSoup(text,'html.parser')
  text = soup.get_text()
  return text
df['review'] = df['review'].apply(clean1)

def clean2(text):
  text = re.sub('[^a-zA-Z]',' ',text)
  return text
df['review'] = df['review'].apply(clean2)

def clean3(text):
  text = text.lower()
  return text
df['review'] = df['review'].apply(clean3)

def clean4(text):
  a = []
  text = text.split()
  for i in text:
    if i not in stopwords.words('english'):
      a.append(i)
  return a
df['review'] = df['review'].apply(clean4)

le = WordNetLemmatizer()
def clean5(text):
  text = [le.lemmatize(word) for word in text]
  text = ' '.join(text)
  return text
df['review'] = df['review'].apply(clean5)

x_train,x_test,y_train,y_test = train_test_split(df['review'],df['sentiment'])

new_x_train = []
new_x_test = []
for i in x_train.index:
  tem = x_train[i]
  new_x_train.append(tem)
for i in x_test.index:
  tem = x_test[i]
  new_x_test.append(tem)
x_train = new_x_train
x_test = new_x_test

st.title('IMBD movie review')
st.subheader('Count Vectorizer')
st.write('This project is based on Linear SVC classifier')
model = Pipeline([('cv',CountVectorizer()),('model',LinearSVC(C=100,max_iter=10000))])
model.fit(x_train,y_train)
message = st.text_area('Enter Text','the movie was good, the actors did a wonderful job, but plot seemed repetitive')
op = model.predict([message])
if st.button('predict'):
  st.title(op)