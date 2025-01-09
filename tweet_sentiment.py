#!/usr/bin/env python
# coding: utf-8

# In[9]:


import io

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

tsv_file_path = 'twitter-2013train-A.tsv'

data = pd.read_csv(tsv_file_path, delimiter='\t')


# #View Dataset

# In[10]:


data.head()


# In[11]:


data.shape


# #Data preprocessing
#  1. Raw data preprocessing
#  2. Counting the type of sentiments (positive,neutral,negative)

# In[12]:


data.category.unique()


# In[13]:


data.isna().sum()


# In[14]:


data[data['category'].isna()]


# In[15]:


# Check if 'clean_text' column exists
if 'clean_text' in data.columns:
    missing_values = data[data['clean_text'].isna()]
    # Handle rows with missing 'clean_text' here
else:
    print("Column 'clean_text' does not exist in DataFrame.")
    # Adjust your approach accordingly


# In[16]:


data[data['clean_text'].isna()]


# In[17]:


data.drop(data[data['clean_text'].isna()].index, inplace=True)
data.drop(data[data['category'].isna()].index, inplace=True)


# In[24]:


tsv_file_path = 'twitter-2013train-A.tsv'
df_train = pd.read_csv(tsv_file_path, delimiter='\t')


# In[25]:


df_train.isnull().sum()


# In[26]:


df_train.head()


# In[27]:


df_train['category'].unique()


# In[28]:


df_train['category'].value_counts()


# In[29]:


df_train = df_train[~df_train['category'].isnull()]


# In[30]:


df_train = df_train[~df_train['clean_text'].isnull()]


# In[31]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df_train['category_1'] = labelencoder.fit_transform(df_train['category'])


# In[32]:


df_train[['category','category_1']].drop_duplicates(keep='first')


# In[33]:


df_train.rename(columns={'category_1':'label'},inplace=True)


# In[36]:


#labeling
reviews = np.array(data['clean_text'])[:]
labels = np.array(data['category'])[:]


# In[37]:


from collections import Counter

Counter(labels)


# Here we remove all the special charecters(@,#,$ etc), punctuations and URL from all the esteemed tweets.
# 
# 
# Next, we have given the token id's to each of the words in the tweet. Then we vectorize all the tokens.

# In[38]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
import csv


def preProcessor(Tweet):
    import re
    from string import punctuation
    text=re.sub(r'(http|ftp|https):\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?', ' ', Tweet)
    text=re.sub(r'['+punctuation+']',' ',Tweet)
    text=re.sub(r'#(\w+)',' ',Tweet)
    text=re.sub(r'@(\w+)',' ',Tweet)
    #print(token.tokenize(text))
    return Tweet

token=RegexpTokenizer(r'\w+')
cv=CountVectorizer(lowercase=True,preprocessor=preProcessor,stop_words='english',ngram_range=(1,1),tokenizer=token.tokenize)
#text_counts=cv.fit_transform(data['Tweet'])
text_counts=cv.fit_transform(data['clean_text'].values.astype('U'))


# Here we split the dataset in test and train part.We use train dataset for training purposes and test dataset for checking the accuracy of our model.

# In[40]:


from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(text_counts,data['sentiment'],test_size=0.3)
x_train, x_test, y_train, y_test = train_test_split(text_counts,data['category'],test_size=0.3)


# #Naive Bayes
# 
#  Naive Bayes classifier is a probabilistic machine learning model that’s used for classification task. The crux of the classifier is based on the Bayes theorem.
# 
# Using Bayes theorem, we can find the probability of A happening, given that B has occurred. Here, B is the evidence and A is the hypothesis. The assumption made here is that the predictors/features are independent. That is presence of one particular feature does not affect the other. Hence it is called naive.

# In[41]:


#Ber_NB
from sklearn.naive_bayes import *
from sklearn import metrics

clf=BernoulliNB()
clf.fit(x_train,y_train)
clf.fit(x_train,y_train)
pred=clf.predict(x_test)
metrics.accuracy_score(y_test, pred)


# #SVM
# Support Vector Machine or SVM is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. However, primarily, it is used for Classification problems in Machine Learning.
# 
# The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.
# 
# SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme cases are called as support vectors, and hence algorithm is termed as Support Vector Machine.

# In[42]:


from sklearn import svm
clf = svm.LinearSVC()
clf.fit(x_train,y_train)
pred=clf.predict(x_test)
metrics.accuracy_score(y_test, pred)


# We can get the better analysis from this confusion matrix.In this matrix if the diagonal elements are higher, the model will be more efficient.

# In[43]:


import matplotlib.pyplot as plt
from sklearn import metrics
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[44]:


#linear
from sklearn.svm import LinearSVC
import sklearn
from sklearn.naive_bayes import *
from sklearn import metrics
from sklearn.metrics import confusion_matrix
clf=LinearSVC()
clf.fit(x_train,y_train)
clf.fit(x_train,y_train)
pred=clf.predict(x_test)
metrics.accuracy_score(y_test, pred)
metrics.accuracy_score(y_test, pred)
cnf_matrix = confusion_matrix(y_test, pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["positive", "negative",'neutral'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["positive", "negative",'neutral'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()
cnf_matrix = confusion_matrix(y_test, pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["positive", "negative",'neutral'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["positive", "negative",'neutral'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()

