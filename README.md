# Internshala-project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv(r'C:\Users\Sanjana Singh\Downloads\train.csv')
test=pd.read_csv(r'C:\Users\Sanjana Singh\Downloads\test.csv')

train.head()

train['subscribed']=train['subscribed'].map({'yes':1,'no':0})

x=train.drop('subscribed',axis=1)
y=train['subscribed']

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3)

from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

logreg.fit(train_x,train_y)

logreg.score(train_x,train_y)

pred=logreg.predict(test_x)
pred

logreg.score(test_x,test_y)

logreg.predict(test)

test1=pd.DataFrame()
test1['subscribed']=logreg.predict(test)
test1.to_csv('test.csv')
test1.head()
test1.shape
test1
