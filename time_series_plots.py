import pandas as pd
from pandasql import sqldf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

### CREATE THE TIME SERIES PLOTS ###
# fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# sns.lineplot(ax=axs[0, 0], x="year", y="danceability", data=initial_load)
# axs[0, 0].set_title('Danceability')
# sns.lineplot(ax=axs[0, 1], x="year", y="energy", data=initial_load)
# axs[0, 1].set_title('Energy')
# sns.lineplot(ax=axs[1, 0], x="year", y="loudness", data=initial_load)
# axs[1, 0].set_title('Loudness')
# sns.lineplot(ax=axs[1, 1], x="year", y="speechiness", data=initial_load)
# axs[1, 1].set_title('Speechiness')

# for ax in axs.flat:
#     ax.set(xlabel='Year')

# fig.tight_layout(pad=3.0)

# plt.savefig("test_plot_6")


### CREATE THE CORRELATION MATRIX ###
audio_features_query =   """
                            SELECT 
                                f.danceability
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
                            FROM 
                                features f
                            """

audio_features_isolated = pysqldf(audio_features_query)

plt.figure(figsize=(12,12))
heatmap = sns.heatmap(audio_features_isolated.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')