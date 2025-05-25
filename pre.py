import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
# data processing


def preprocess():
    df= pd.read_csv("movie_metadata.csv")
    #drop useless(we think it's) columns
    df=df.drop(columns=["color","director_name","actor_1_name","actor_2_name","actor_3_name","language","country","movie_imdb_link","plot_keywords","movie_title"])
    #drop row with missing(NAN)value
    df=df.dropna()
    # count how many movies (row )left
    df.count()

    #encode genres(multi-label)
    df['genres']=df['genres'].str.split('|')
    mlb= MultiLabelBinarizer()
    genre_encoded=pd.DataFrame(mlb.fit_transform(df['genres']), columns=mlb.classes_, index=df.index)
    df=df.drop(columns=["genres"]).join(genre_encoded)

    # encode content_rating
    le=LabelEncoder()
    df['content_rating'] = le.fit_transform(df['content_rating'])

    #the value we want to predict
    scores=df["imdb_score"].values
    df= df.drop(columns=["imdb_score"])
    return df,scores

