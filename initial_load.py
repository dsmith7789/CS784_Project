import pandas as pd
from pandasql import sqldf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

pysqldf = lambda q: sqldf(q, globals())

charts = pd.read_csv("charts_with_spotify_ids_final.csv")
# change week column to datetime: https://www.geeksforgeeks.org/convert-the-column-type-from-string-to-datetime-format-in-pandas-dataframe/#
charts['given_date']= pd.to_datetime(charts['given_date'])

features = pd.read_csv("track_features.csv")
popularity = pd.read_csv("track_popularity.csv")

initial_sql_query = """
                    SELECT 
                        c.track_id "track_id"
                        , c.given_date "week"
                        , CAST(SUBSTR(given_date, 1, 4) AS integer) "year"
                        , c.given_rank "rank"
                        , c.given_peak_rank "peak_rank"
                        , c.given_weeks_on_board "weeks_on_board"
                        , f.danceability
                        , f.energy
                        , f.key
                        , f.loudness
                        , f.mode
                        , f.speechiness
                        , f.acousticness
                        , f.instrumentalness
                        , f.liveness
                        , f.valence
                        , f.tempo
                        , f.duration_ms
                        , f.time_signature
                        , p.popularity
                    FROM 
                        charts c
                        INNER JOIN features f ON c.track_id = f.id
                        INNER JOIN popularity p on f.id = p.id
                    """

initial_load = pysqldf(initial_sql_query)
print(initial_load)
