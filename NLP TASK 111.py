#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

The Data
Read the data
# In[3]:


df=pd.read_csv('nlp1.test.csv')
df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.duplicated().sum()


# In[7]:


df.describe()


# In[8]:


df.nunique()


# In[10]:


df.value_counts


# Create a new column called "text length" which is the number of words in the text column.

# In[11]:


df["text length"] = df["target_explanation_english"].apply(len)


# In[12]:


df.head()

EDA
Let's explore the data

Imports
Import the data visualization libraries if you haven't done so already.Create a boxplot of text length for each star company.
# In[18]:


sns.boxplot(x="company_name", y="text length",color='red',data=df)


# Create a countplot of the number of occurrences for each type of star rating.

# # NLP Classification Task
Let's move on to the actual task. To make things a little easier, go ahead and only grab reviews that were either 1 star or 5 stars.

Create a dataframe called df_class that contains the columns of df dataframe but for only the 1 or 5 star reviews.
# In[20]:


X = df["target_explanation_english"] 
y = df["company_name"]


# Import CountVectorizer and create a CountVectorizer object.

# In[21]:


from sklearn.feature_extraction.text import CountVectorizer


# In[22]:


cv = CountVectorizer()


# Use the fit_transform method on the CountVectorizer object and pass in X (the 'target_explanation_english' column). Save this result by overwriting X.

# In[24]:


X = cv.fit_transform(X)

Train Test Split
Let's split our data into training and testing data.

Use train_test_split to split up the data into X_train, X_test, y_train, y_test. Use test_size=0.3 and random_state=101
# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

Training a Model
Time to train a model!

Import MultinomialNB and create an instance of the estimator and call is nb
# In[27]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

Now fit nb using the training data.
# In[28]:


nb.fit(X_train, y_train)


# # Predictions and Evaluations
# Time to see how our model did!
# 
# Use the predict method off of nb to predict labels from X_test.

# In[29]:


predictions = nb.predict(X_test)

Create a confusion matrix and classification report using these predictions and y_test
# In[30]:


from sklearn.metrics import confusion_matrix, classification_report


# In[31]:


print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

Using Text Processing
Import TfidfTransformer from sklearn.
# In[33]:


from sklearn.feature_extraction.text import TfidfTransformer

Import Pipeline from sklearn.
# In[34]:


from sklearn.pipeline import Pipeline

Now create a pipeline with the following steps:CountVectorizer(), TfidfTransformer(),MultinomialNB()
# In[35]:


pipeline = Pipeline([
    ("bow", CountVectorizer()),
    ("tfidf", TfidfTransformer()),
    ("model", MultinomialNB())
])

Using the Pipeline
Time to use the pipeline! Remember this pipeline has all your pre-process steps in it already, meaning we'll need to re-split the original data (Remember that we overwrote X as the CountVectorized version. What we need is just the textTrain Test Split
Redo the train test split on the df .
# In[37]:


X = df["target_explanation_english"]
y = df["company_name"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

Now fit the pipeline to the training data. Remember you can't use the same training data as last time because that data has already been vectorized. We need to pass in just the text and labels
# In[38]:


pipeline.fit(X_train, y_train)

Predictions and Evaluation
Now use the pipeline to predict from the X_test and create a classification report and confusion matrix. You should notice strange results.
# In[39]:


new_pred = pipeline.predict(X_test)


# In[40]:


print(confusion_matrix(y_test, new_pred))
print(classification_report(y_test, new_pred))


# In[ ]:




