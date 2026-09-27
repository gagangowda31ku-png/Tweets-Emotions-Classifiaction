import pandas as pd
import nltk
import warnings
import os
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
nltk.download('punkt')
nltk.download('stopwords')

warnings.filterwarnings("ignore", category=UserWarning)

try:
    train_df = pd.read_csv('data/raw/train.csv')
    test_df = pd.read_csv('data/raw/test.csv')
    english_stopwords = stopwords.words("english")
    stemmer = SnowballStemmer(language = 'english')

    def tokenize(text):
        return [stemmer.stem(token) for token in word_tokenize(text) if token.lower().isalpha() and token not in english_stopwords]

    vectorizer = TfidfVectorizer(lowercase = True , 
                             tokenizer = tokenize,
                             stop_words = english_stopwords , 
                             max_features = 1500 , 
                             ngram_range = (1,2))
    inputs = vectorizer.fit_transform(train_df.content.values)
    test_input = vectorizer.transform(test_df.content.values)
    train_data = pd.DataFrame(inputs.toarray())
    test_data = pd.DataFrame(test_input.toarray())
    train_data['sentiment'] = train_df[['sentiment']]
    test_data['sentiment']= test_df[['sentiment']]
    

except Exception as e:
    print("Error : An Unexpected error occured while preprocessing the data")
    print(f"Error: {e}")
    raise

data_path = os.path.join('data' , 'processed')
os.makedirs(data_path , exist_ok = True)
train_data.to_csv(os.path.join(data_path , 'train_preprocess.csv'),index = False)
test_data.to_csv(os.path.join(data_path , 'test_preprocess.csv'),index =False)
