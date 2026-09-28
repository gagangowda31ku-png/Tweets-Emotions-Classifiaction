import pandas as pd 
import numpy as np 
import pickle
from sklearn.metrics import classification_report , accuracy_score 
from scipy.sparse import csr_matrix

from sklearn.naive_bayes import MultinomialNB



train_df = pd.read_csv('data/processed/train_preprocess.csv')

x_train = train_df.iloc[:,0:-1].values
y_train = train_df.iloc[: , -1].values

model = MultinomialNB(alpha = 0.7 , fit_prior = True ,  force_alpha = True)
model.fit(x_train,y_train)

pickle.dump(model , open('model.pkl' , 'wb'))



    