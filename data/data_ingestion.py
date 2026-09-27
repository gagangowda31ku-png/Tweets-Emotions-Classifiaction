import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


def load_data(data_url: str)-> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        return df

    except pd.errors.ParserError as e:
        print("Error : Cannot load the CSV file from the given data ")
        print(e)
        raise
    except Exception as e:
        print("ERROR: An unexpected error occured while loading the file")
        print(e)
        raise

def preprocess_load(df:pd.DataFrame)-> pd.DataFrame:
    try:
        df.drop(columns = ['tweet_id'],inplace = True)
  
        train_df , test_df = train_test_split(df , test_size = 0.2 , random_state = 42)
        categories = [[
        'anger',
        'hate',
        'worry',
        'sadness',
        'boredom',
        'empty',
        'neutral',
        'relief',
        'surprise',
        'enthusiasm',
        'fun',
        'happiness',
        'love'
        ]]
        encoder = OrdinalEncoder(categories = categories)
        train_df[['sentiment']] = encoder.fit_transform(train_df[['sentiment']])
        test_df[['sentiment']] = encoder.transform(test_df[['sentiment']])
        

        return train_df , test_df

    except KeyError as e:
        print("Error : Missing column {e} in the DataFrame")
        print(e)
        raise
    except Exception as e:
        print("Error : An unexpected Error occured while prepeocessing the file")
        print(e)
        raise

def save_data(train_df: pd.DataFrame , test_df: pd.DataFrame , data_path:str)-> None:
    try:
        data_path = os.path.join(data_path , 'raw')
        os.makedirs(data_path , exist_ok = True)
        train_df.to_csv(os.path.join(data_path , 'train.csv'),index = False)
        test_df.to_csv(os.path.join(data_path , 'test.csv'),index =False)
    except Exception as e:
        print("Error : An unexpected error occured while saving the Data ")
        print(e)
        raise

def main_data():
    try:

        df = load_data("https://raw.githubusercontent.com/entbappy/Branching-tutorial/refs/heads/master/tweet_emotions.csv")
        train_df , test_df = preprocess_load(df)
        save_data(train_df , test_df , 'data')

    except Exception as e:
        print("Error : An unexpected error occured while saving the Data ")
        print(e)
        raise

main_data()




