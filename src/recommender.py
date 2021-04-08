import logging
import numpy as np
import pandas as pd
from pyspark.ml.recommendation import ALS
import pyspark
from pyspark.sql import SparkSession


class MovieRecommender():
    spark = SparkSession.builder.getOrCreate()
    """Template class for a Movie Recommender system."""

    def __init__(self):
        """Constructs a MovieRecommender"""
        self.logger = logging.getLogger('reco-cs')
        #self.neighborhood_size = neighborhood_size
        self.trained_model = None
        
        self.init_model = ALS(
                    itemCol='movie',
                    userCol='user',
                    ratingCol='rating',
                    nonnegative=True,    
                    regParam=0.1,
                    rank=10) 
    
    def fit(self, ratings):
        spark = MovieRecommender.spark
        spark_df = spark.createDataFrame(ratings) 
        self.trained_model = self.init_model.fit(spark_df)
        
        """
        Trains the recommender on a given set of ratings.

        Parameters
        ----------
        ratings : pandas dataframe, shape = (n_ratings, 4)
                  with columns 'user', 'movie', 'rating', 'timestamp'

        Returns
        -------
        self : object
            Returns self.
        """
        self.logger.debug("starting fit")

        # ...

        self.logger.debug("finishing fit")
        return(self)


    def transform(self, requests):
        """
        Predicts the ratings for a given set of requests.

        Parameters
        ----------
        requests : pandas dataframe, shape = (n_ratings, 2)
                  with columns 'user', 'movie'

        Returns
        -------
        dataframe : a pandas dataframe with columns 'user', 'movie', 'rating'
                    column 'rating' containing the predicted rating
        """
        spark = MovieRecommender.spark
        sparked_input = spark.createDataFrame(requests)
        sparked_predicted = recommender.transform(sparked_input)

        #requests['rating'] = np.random.choice(range(1, 5), requests.shape[0])
        self.logger.debug("finishing predict")
        self.logger.debug("starting predict")
        self.logger.debug("request count: {}".format(requests.shape[0]))

        return sparked_predicted.toPandas()
        
       


if __name__ == "__main__":
    logger = logging.getLogger('reco-cs')
    logger.critical('you should use run.py instead')
