
import pandas as pd


ratings_df = pd.read_csv('../data/training.csv')

movies_df = pd.read_csv("data/movies.dat", sep='::', names=["num", "title", "genres"])
movies_df['year'] = movies_df['title'].str[-5:-1]
movies_df['title'] = movies_df['title'].str[:-7]
movies_df['genres'] = movies_df['genres'].str.split('|')
movies_df.drop('num', axis=1, inplace=True)


users_df = pd.read_csv("data/users.dat", sep='::', names=["num", "gender", "age", 'number', 'id'])
users_df['gender'] = users_df['gender'].map({'F': 1, 'M': 0})
users_df.drop(['num', 'id'], axis=1, inplace=True)


#merging movie_metadata with movies_df 
movie_metadata = pd.read_csv('data/movies_metadata.csv')
movies_metadata_df = movies_df.merge(movie_metadata, left_on='title', right_on='title', how='left')
movies_metadata_df['title']=movies_metadata_df['title'].apply(lambda x: x.split(',')[1].lstrip(' ') + " "+ x.split(',')[0] if ', The' in x else x)