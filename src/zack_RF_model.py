import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np

movies_df = pd.read_csv("../data/movies.dat", sep='::', names=["request_movie_id", "title", "genres"])
movies_df['year'] = movies_df['title'].str[-5:-1]
movies_df['title'] = movies_df['title'].str[:-7]
movies_df['genres'].fillna('[Other]', inplace=True)

movies_df['genres'] = movies_df['genres'].str.split('|')
movies_df['title']=movies_df['title'].apply(lambda x: x.split(',')[1].lstrip(' ') + " "+ x.split(',')[0] if ', The' in x else x)

users_df = pd.read_csv("../data/users.dat", sep='::', names=["request_id", "gender", "age", 'number', 'id'])
users_df['gender'] = users_df['gender'].map({'F': 1, 'M': 0})

#merging movie_metadata with movies_df
movie_metadata = pd.read_csv('../data/movies_metadata.csv', low_memory=False)
movies_metadata_df = movies_df.merge(movie_metadata, left_on='title', right_on='title', how='left')
movies_metadata_df = movies_metadata_df.drop_duplicates(subset='title', keep='first')

training_df = pd.read_csv('../data/training.csv')


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

RF_df = training_df.copy()
RF_df = RF_df.merge(users_df, how='left', left_on='user', right_on='request_id')
RF_df = RF_df.merge(movies_metadata_df, how='left', left_on='movie', right_on='request_movie_id')

RF_df['genres_x'].fillna(0, inplace=True)
genres = list(RF_df['genres_x'])
genres = [x if x else ['Other'] for x in genres]
RF_df['genres'] = genres
mlb.fit(RF_df['genres'])
RF_df[mlb.classes_] = mlb.transform(RF_df['genres'])

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.utils import parallel_backend

X = RF_df.copy()
X.drop(['user', 'movie', 'timestamp', 'request_id', 'genres_x', 'genres', 'id_x', 'request_movie_id', 'title', 'genres_x', 'adult', 'belongs_to_collection',
         'genres_y', 'homepage', 'id_y', 'imdb_id', 'original_language', 'original_title', 'overview', 'poster_path',
         'production_companies', 'production_countries', 'release_date', 'spoken_languages', 'status', 'tagline', 'video'], axis=1, inplace=True)
X.fillna(X.median(), inplace=True)
y = X.pop('rating')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


RF_model = RandomForestRegressor(max_depth=10, max_features=0.4, n_estimators=800,
                                min_samples_split=5, min_samples_leaf=4, n_jobs=8)

RF_model.fit(X, y)


# y_pred = RF_model.predict(X_test)
# rmse = (mean_squared_error(y_test, y_pred)**0.5)
# print(rmse)
# print(RF_model.feature_importances_)

from joblib import dump, load
dump(RF_model, '../data/RF_model.joblib')

# grid = {
#         'n_estimators': [800],
#         'max_depth': [10],
#         'min_samples_split': [5],
#         'min_samples_leaf': [4],
#         'max_features': [0.4],
#         }
#
# with parallel_backend('multiprocessing'):
#     model = RandomForestRegressor()
#     RF_gridsearch = GridSearchCV(estimator=model, param_grid=grid,
#                                  cv=5, verbose=0, n_jobs=8)
#     RF_gridsearch.fit(X_train[0:10000], y_train[0:10000])
#     best_model = RF_gridsearch.best_estimator_
#     best_score = RF_gridsearch.best_score_
#     best_params = RF_gridsearch.best_params_
#     print(f'score: {best_score}, params: {best_params}')