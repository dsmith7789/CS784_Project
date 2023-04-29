import pandas as pd
from pandasql import sqldf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import plotly.express as px
# from yellowbrick.cluster import KElbowVisualizer
# from itertools import permutations

pysqldf = lambda q: sqldf(q, globals())

#main_df = pd.read_csv("charts_with_features_id_and_popularity.csv")
# change week column to datetime: https://www.geeksforgeeks.org/convert-the-column-type-from-string-to-datetime-format-in-pandas-dataframe/#
#main_df['week']= pd.to_datetime(main_df['week'])

features = pd.read_csv("track_features.csv")
popularity = pd.read_csv("track_popularity.csv")


### 1. CREATE THE TIME SERIES PLOTS ###
# fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# sns.lineplot(ax=axs[0, 0], x="year", y="danceability", data=main_df)
# axs[0, 0].set_title('Danceability')
# sns.lineplot(ax=axs[0, 1], x="year", y="energy", data=main_df)
# axs[0, 1].set_title('Energy')
# sns.lineplot(ax=axs[1, 0], x="year", y="loudness", data=main_df)
# axs[1, 0].set_title('Loudness')
# sns.lineplot(ax=axs[1, 1], x="year", y="speechiness", data=main_df)
# axs[1, 1].set_title('Speechiness')

# for ax in axs.flat:
#     ax.set(xlabel='Year')

# fig.tight_layout(pad=3.0)

# plt.savefig("test_plot_6")


### 2. CREATE THE CORRELATION MATRIX ###
audio_features_query =   """
                            SELECT
                                f.id 
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
                            FROM 
                                features f
                            """
audio_features = pysqldf(audio_features_query)
# audio_features_isolated = audio_features.drop('id', axis=1)
audio_features_isolated = audio_features[["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "valence", "tempo"]]

# plt.figure(figsize=(12,12))
# heatmap = sns.heatmap(audio_features_isolated.corr(method='spearman'), vmin=-1, vmax=1, annot=True, cmap='BrBG')
# heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
# plt.savefig('heatmap_spearman.png', dpi=300, bbox_inches='tight')

### 3. CLUSTERING ###

# determine ideal number of clusters (seems to be 3-4 per the elbow plot, will use 4)
x = audio_features_isolated.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
scaled_audio_features = pd.DataFrame(x_scaled)
scaled_audio_features.columns = audio_features_isolated.columns
print(scaled_audio_features)
# plt.figure(figsize=(12,12))
# heatmap = sns.heatmap(scaled_audio_features.corr(method='spearman'), vmin=-1, vmax=1, annot=True, cmap='BrBG')
# heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
# plt.savefig('normalized_heatmap_spearman.png', dpi=300, bbox_inches='tight')

inertia = []
k_vals = []
K = np.arange(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k, n_init="auto").fit(scaled_audio_features)
    kmeanModel.fit(scaled_audio_features)
    inertia.append(kmeanModel.inertia_)
    k_vals.append(k)

plt.plot(k_vals, inertia)
plt.title("Elbow Plot")
plt.savefig("elbow_plot")

# Cluster the songs into 4 different clusters
KMeans = KMeans(n_clusters=4).fit(scaled_audio_features)
audio_features['cluster'] = KMeans.fit_predict(scaled_audio_features)
scaled_audio_features['cluster'] = audio_features['cluster']
scaled_audio_features['id'] = audio_features['id']
tracks_clustered = audio_features.reset_index(drop=True)
scaled_tracks_clustered = scaled_audio_features.reset_index(drop=True)
print(tracks_clustered)
print(scaled_tracks_clustered)
tracks_clustered.to_csv('tracks_clustered.csv', encoding='utf-8', index=False)
scaled_tracks_clustered.to_csv('scaled_tracks_clustered.csv', encoding='utf-8', index=False)

### 4. CLUSTER CHARACTERISTICS ###
# Join the scaled features to the ids and clusters
polar_cluster0_query =  """
                        SELECT
                            AVG(speechiness)
                            , AVG(energy)
                            , AVG(danceability)
                            , AVG(tempo)
                            , AVG(instrumentalness)
                            , AVG(acousticness)
                            , AVG(valence)
                            , AVG(loudness)
                        FROM
                            scaled_tracks_clustered
                        WHERE
                            cluster = 0
                        """
polar_cluster0_df = pysqldf(polar_cluster0_query)
print(polar_cluster0_df)
df = pd.DataFrame(dict(
    r=polar_cluster0_df.values.tolist()[0],
    theta=['speechiness','energy','danceability', 'tempo', 'instrumentalness', 'acousticness', 'valence', 'loudness']))
fig = px.line_polar(df, r='r', theta='theta', line_close=True)
fig.update_traces(fill='toself')
fig.write_image("cluster0_polar.png")

polar_cluster1_query =  """
                        SELECT
                            AVG(speechiness)
                            , AVG(energy)
                            , AVG(danceability)
                            , AVG(tempo)
                            , AVG(instrumentalness)
                            , AVG(acousticness)
                            , AVG(valence)
                            , AVG(loudness)
                        FROM
                            scaled_tracks_clustered
                        WHERE
                            cluster = 1
                        """
polar_cluster1_df = pysqldf(polar_cluster1_query)
print(polar_cluster1_df)
df = pd.DataFrame(dict(
    r=polar_cluster1_df.values.tolist()[0],
    theta=['speechiness','energy','danceability', 'tempo', 'instrumentalness', 'acousticness', 'valence', 'loudness']))
fig = px.line_polar(df, r='r', theta='theta', line_close=True)
fig.update_traces(fill='toself')
fig.write_image("cluster1_polar.png")

polar_cluster2_query =  """
                        SELECT
                            AVG(speechiness)
                            , AVG(energy)
                            , AVG(danceability)
                            , AVG(tempo)
                            , AVG(instrumentalness)
                            , AVG(acousticness)
                            , AVG(valence)
                            , AVG(loudness)
                        FROM
                            scaled_tracks_clustered
                        WHERE
                            cluster = 2
                        """
polar_cluster2_df = pysqldf(polar_cluster2_query)
df = pd.DataFrame(dict(
    r=polar_cluster2_df.values.tolist()[0],
    theta=['speechiness','energy','danceability', 'tempo', 'instrumentalness', 'acousticness', 'valence', 'loudness']))
print(polar_cluster2_df)
fig = px.line_polar(df, r='r', theta='theta', line_close=True)
fig.update_traces(fill='toself')
fig.write_image("cluster2_polar.png")

polar_cluster3_query =  """
                        SELECT
                            AVG(speechiness)
                            , AVG(energy)
                            , AVG(danceability)
                            , AVG(tempo)
                            , AVG(instrumentalness)
                            , AVG(acousticness)
                            , AVG(valence)
                            , AVG(loudness)
                        FROM
                            scaled_tracks_clustered
                        WHERE
                            cluster = 3
                        """
polar_cluster3_df = pysqldf(polar_cluster3_query)
print(polar_cluster3_df)
df = pd.DataFrame(dict(
    r=polar_cluster3_df.values.tolist()[0],
    theta=['speechiness','energy','danceability', 'tempo', 'instrumentalness', 'acousticness', 'valence', 'loudness']))
fig = px.line_polar(df, r='r', theta='theta', line_close=True)
fig.update_traces(fill='toself')
fig.write_image("cluster3_polar.png")