import pickle 
import pandas as pd
import json
from sklearn.metrics import accuracy_score ,  precision_score , recall_score , f1_score

model = pickle.load(open('model.pkl' , 'rb'))

test_df = pd.read_csv('data/processed/test_preprocess.csv')

x_test = test_df.iloc[:,0:-1].values
y_test = test_df.iloc[: , -1].values

y_pred = model.predict(x_test)

accuracy =  accuracy_score(y_test, y_pred)
precision =  precision_score(y_test, y_pred , average = 'weighted' , zero_division = 0)
recall =  recall_score(y_test, y_pred , average = 'weighted' , zero_division = 0)
f1_Score =  f1_score(y_test, y_pred , average = 'weighted' , zero_division = 0)

metrics = {
    "accuary": accuracy,
    'precision': precision, 
    'recall': recall,
    'f1_score': f1_Score 
}

with open('metrics.json' , 'w') as file:
    json.dump(metrics , file , indent = 4)

    