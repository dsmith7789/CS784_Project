import pandas as pd
from pandasql import sqldf
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mlp
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import plotly.express as px
# from yellowbrick.cluster import KElbowVisualizer
# from itertools import permutations

pysqldf = lambda q: sqldf(q, globals())
local_pysqldf = lambda q: sqldf(q, locals())    # for locally defined tables in a function
main_df = pd.read_csv("charts_with_features_id_and_popularity.csv")
# change week column to datetime: https://www.geeksforgeeks.org/convert-the-column-type-from-string-to-datetime-format-in-pandas-dataframe/#
main_df['week']= pd.to_datetime(main_df['week'])

features = pd.read_csv("track_features.csv")
popularity = pd.read_csv("track_popularity.csv")

def time_series_plots():
    ### 1. CREATE THE TIME SERIES PLOTS ###
    mlp.rcParams['font.size'] = 16
    fig, axs = plt.subplots(2, 5, figsize=(45, 20))


    sns.lineplot(ax=axs[0, 0], x="year", y="danceability", data=main_df)
    axs[0, 0].set_title('Danceability', fontsize=36)
    print("Plotted Danceability")
    sns.lineplot(ax=axs[0, 1], x="year", y="energy", data=main_df)
    axs[0, 1].set_title('Energy', fontsize=36)
    print("Plotted Energy")
    sns.lineplot(ax=axs[0, 2], x="year", y="loudness", data=main_df)
    axs[0, 2].set_title('Loudness', fontsize=36)
    print("Plotted Loudness")
    sns.lineplot(ax=axs[0, 3], x="year", y="speechiness", data=main_df)
    axs[0, 3].set_title('Speechiness', fontsize=36)
    print("Plotted Speechiness")
    sns.lineplot(ax=axs[0, 4], x="year", y="acousticness", data=main_df)
    axs[0, 4].set_title('Acousticness', fontsize=36)
    print("Plotted Acousticness")
    sns.lineplot(ax=axs[1, 0], x="year", y="instrumentalness", data=main_df)
    axs[1, 0].set_title('Instrumentalness', fontsize=36)
    print("Plotted Instrumentalness")
    sns.lineplot(ax=axs[1, 1], x="year", y="liveness", data=main_df)
    axs[1, 1].set_title('Liveness', fontsize=36)
    print("Plotted Liveness")
    sns.lineplot(ax=axs[1, 2], x="year", y="valence", data=main_df)
    axs[1, 2].set_title('Valence', fontsize=36)
    print("Plotted Valence")
    sns.lineplot(ax=axs[1, 3], x="year", y="tempo", data=main_df)
    axs[1, 3].set_title('Tempo', fontsize=36)
    print("Plotted Tempo")
    sns.lineplot(ax=axs[1, 4], x="year", y="duration_ms", data=main_df)
    axs[1, 4].set_title('Duration', fontsize=36)
    print("Plotted Duration")

    for ax in axs.flat:
        ax.set(xlabel='Year')

    fig.tight_layout(pad=10)
    fig.suptitle("Audio Features Over Time", fontsize=48)

    #plt.xticks(fontsize=25)

    plt.savefig("time_series_graphs")

def create_histograms(start_year, end_year):
    histogram_query = f"""
                        SELECT
                            *
                        FROM
                            main_df 
                        WHERE
                            year BETWEEN {start_year} AND {end_year}
                        """
    histogram_df = pysqldf(histogram_query)

    fig, axs = plt.subplots(2, 6, figsize=(45, 20))

    # Plot the data
    sns.histplot(ax=axs[0, 0], data=histogram_df, x="danceability", kde=True)
    print("Plotted Danceability")
    sns.histplot(ax=axs[0, 1], data=histogram_df, x="energy", kde=True)
    print("Plotted Energy")
    sns.histplot(ax=axs[0, 2], data=histogram_df, x="key", kde=True)
    print("Plotted Key")
    sns.histplot(ax=axs[0, 3], data=histogram_df, x="loudness", kde=True)
    print("Plotted Loudness")
    sns.histplot(ax=axs[0, 4], data=histogram_df, x="speechiness", kde=True)
    print("Plotted Speechiness")
    sns.histplot(ax=axs[0, 5], data=histogram_df, x="acousticness", kde=True)
    print("Plotted Acousticness")
    sns.histplot(ax=axs[1, 0], data=histogram_df, x="liveness", kde=True)
    print("Plotted Liveness")
    sns.histplot(ax=axs[1, 1], data=histogram_df, x="valence", kde=True)
    print("Plotted Valence")
    sns.histplot(ax=axs[1, 2], data=histogram_df, x="tempo", kde=True)
    print("Plotted Tempo")
    sns.histplot(ax=axs[1, 3], data=histogram_df, x="duration_ms", kde=True)
    print("Plotted Duration")
    sns.histplot(ax=axs[1, 4], data=histogram_df, x="popularity", kde=True)
    print("Plotted Popularity")

    print("Plotted Data...")

    # Appropriate Titles
    axs[0, 0].set_title('Danceability', fontsize=36)
    axs[0, 1].set_title('Energy', fontsize=36)
    axs[0, 2].set_title('Key', fontsize=36)
    axs[0, 3].set_title('Loudness', fontsize=36)
    axs[0, 4].set_title('Speechiness', fontsize=36)
    axs[0, 5].set_title('Acousticness', fontsize=36)
    axs[1, 0].set_title('Liveness', fontsize=36)
    axs[1, 1].set_title('Valence', fontsize=36)
    axs[1, 2].set_title('Tempo', fontsize=36)
    axs[1, 3].set_title('Duration (ms)', fontsize=36)
    axs[1, 4].set_title('Current Popularity', fontsize=36)
    print("Titled Plots...")

    # delete unneeded plot
    fig.delaxes(axs[1, 5])

    fig.tight_layout(pad=10)
    fig.suptitle(f"Audio Features, {start_year} to {end_year}", fontsize=48)

    print("Saving Figure...")
    plt.savefig(f"histograms_{start_year}_to_{end_year}")
    pass

def generate_basic_descriptive_statistics(start_year, end_year):
    sql_query = f"""
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
                            main_df f
                        WHERE
                            year BETWEEN {start_year} AND {end_year}
                        """
    dataset = pysqldf(sql_query)
    mean_df = dataset.mean()
    print(mean_df)
    median_df = dataset.median()
    print(median_df)
    std_df = dataset.std()
    print(std_df)

def create_correlation_matrix():
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
                                    , p.popularity
                                FROM 
                                    features f
                                    INNER JOIN popularity p on f.id = p.id
                                """
    audio_features = pysqldf(audio_features_query)
    audio_features_isolated = audio_features.drop('id', axis=1)
    # audio_features_isolated = audio_features[["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "valence", "tempo"]]

    plt.figure(figsize=(12,12))
    heatmap = sns.heatmap(audio_features_isolated.corr(method='spearman'), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
    plt.savefig('heatmap_spearman_2.png', dpi=300, bbox_inches='tight')
    pass

def find_ideal_number_clusters():
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
                                FROM 
                                    features f
                                """
    audio_features = pysqldf(audio_features_query)
    audio_features_isolated = audio_features.drop('id', axis=1)

    # determine ideal number of clusters (seems to be 3-4 per the elbow plot, will use 4)
    x = audio_features_isolated.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    scaled_audio_features = pd.DataFrame(x_scaled)
    scaled_audio_features.columns = audio_features_isolated.columns
    print(scaled_audio_features)

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

def add_clusters():
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
                                FROM 
                                    features f
                                """
    audio_features = pysqldf(audio_features_query)
    audio_features_isolated = audio_features.drop('id', axis=1)

    # scale the audio features
    x = audio_features_isolated.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    scaled_audio_features = pd.DataFrame(x_scaled)
    scaled_audio_features.columns = audio_features_isolated.columns

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

def characterize_clusters(cluster_num):
    scaled_tracks_clustered = pd.read_csv('scaled_tracks_clustered.csv')
    print(scaled_tracks_clustered)
    polar_cluster_query =  f"""
                            SELECT
                                AVG(energy)
                                , AVG(danceability)
                                , AVG(tempo)
                                , AVG(acousticness)
                                , AVG(valence)
                                , AVG(loudness)
                            FROM
                                scaled_tracks_clustered
                            WHERE
                                cluster = {cluster_num}
                            """
    polar_cluster_df = sqldf(polar_cluster_query, locals()) # because it's not a globally defined table
    print(polar_cluster_df)
    df = pd.DataFrame(dict(
        r=polar_cluster_df.values.tolist()[0],
        theta=['energy','danceability', 'tempo', 'acousticness', 'valence', 'loudness']))
    fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    fig.update_traces(fill='toself')
    fig.write_image(f"cluster_{cluster_num}_polar.png")

def correlate_features_with_curr_popularity():
    query = """
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
                , p.popularity
            FROM 
                features f
                INNER JOIN popularity p on f.id = p.id
            """
    df = sqldf(query)
    features = list(df.columns)
    features.remove("popularity")
    correlations = []
    for feature in features:
        correlations.append(df[feature].corr(df['popularity']))

    plt.figure(figsize=(12,12))
    heatmap = sns.heatmap(correlations, vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
    plt.savefig('feature_popularity_heatmap.png', dpi=300, bbox_inches='tight')

def get_median_popularity(lower_bound):
    # 28487 different songs in the dataset
    sql_query = f"""
                SELECT 
                    p.popularity
                FROM
                    popularity p
                WHERE 
                    p.popularity > {lower_bound}
                ORDER BY
                    p.popularity
                LIMIT
                    1
                OFFSET (
                    SELECT 
                        COUNT(*) 
                    FROM 
                        popularity) / 2
                """
    df = sqldf(sql_query)
    print(df)
    median = df['popularity'].iloc[0]
    print(median)
    return median

def append_popular_label():
    median_popularity = get_median_popularity(lower_bound=5)
    q = f"""
        SELECT
            p.*
            , CASE
                WHEN p.popularity > {median_popularity} THEN 1
                ELSE 0
                END AS "is_popular"
        FROM
            popularity p
    """
    df = sqldf(q)
    print(df)
    df.to_csv('track_popularity_binary_classification.csv', encoding='utf-8', index=False)

def aic():
    import statsmodels.api as sm

    # put the dataframe together
    feature_table = pd.read_csv("track_features.csv")
    popularity_labeled = pd.read_csv("track_popularity_binary_classification.csv")
    q = f"""
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
            , pl.is_popular
        FROM
            feature_table f
            INNER JOIN popularity_labeled pl on f.id = pl.id
    """
    df = sqldf(q, locals()) # because it's not a globally defined table
    feature_list = ['danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
    print(feature_list)
    X = df[feature_list]
    y = df['is_popular']

    # split X and y into training and testing sets
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)
    
    # building the model and fitting the data
    log_reg = sm.Logit(y_train, X_train).fit()

    print(log_reg.summary())


def get_important_features_for_popularity_logistic_regression(num_features):
    print(f"choosing best {num_features} features")
    # put the dataframe together
    feature_table = pd.read_csv("track_features.csv")
    popularity_labeled = pd.read_csv("track_popularity_binary_classification.csv")
    q = f"""
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
            , pl.is_popular
        FROM
            feature_table f
            INNER JOIN popularity_labeled pl on f.id = pl.id
    """
    df = sqldf(q, locals()) # because it's not a globally defined table
    feature_list = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
    print(feature_list)
    X = df[feature_list]
    y = df['is_popular']


    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression

    # Create a logistic regression model
    model = LogisticRegression()

    # Use RFE to select the top 10 features
    rfe = RFE(model, n_features_to_select=num_features)
    rfe.fit(X, y)

    # Print the selected features
    print(rfe.support_)

def logistic_regression_popularity():
    # put the dataframe together
    feature_table = pd.read_csv("track_features.csv")
    popularity_labeled = pd.read_csv("track_popularity_binary_classification.csv")
    q = f"""
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
            , pl.is_popular
        FROM
            feature_table f
            INNER JOIN popularity_labeled pl on f.id = pl.id
    """
    df = sqldf(q, locals()) # because it's not a globally defined table
    #feature_list = ['danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
    feature_list = ['loudness', 'acousticness', 'energy', 'speechiness', 'valence', 'danceability', 'instrumentalness', 'mode', 'time_signature', 'liveness', 'tempo', 'key'] # Chosen based on RFE until we start getting some 1 classifications
    X = df[feature_list]
    print(X)
    y = df['is_popular']
    print(y)

    # roc curve for logistic regression model with optimal threshold
    from numpy import sqrt
    from numpy import argmax
    from sklearn import metrics
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve
    from matplotlib import pyplot

    # split into train/test sets
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.25, random_state=16)
    # fit a model
    model = LogisticRegression()
    model.fit(trainX, trainy)
    # predict probabilities
    yhat = model.predict_proba(testX)
    # keep probabilities for the positive outcome only
    yhat = yhat[:, 1]
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(testy, yhat)
    # calculate the g-mean for each threshold
    gmeans = sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    print('AUC=%.3f' % (metrics.auc(fpr, tpr)))
    # plot the roc curve for the model
    pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
    pyplot.plot(fpr, tpr, marker='.', label='Logistic')
    pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    # show the plot
    pyplot.savefig("log_reg_plot")
    
    threshold = 0.45
    y_hat = (yhat > threshold).astype('float')
    cnf_matrix = metrics.confusion_matrix(testy, y_hat)
    print(cnf_matrix)

def popularity_correlations():
    # put the dataframe together
    q = f"""
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
            , p.popularity
        FROM
            features f
            INNER JOIN popularity p on f.id = p.id
    """
    df = pysqldf(q)
    feature_list = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
    
    ### CREATE THE SCATTER PLOTS ###
    mlp.rcParams['font.size'] = 16
    fig, axs = plt.subplots(2, 5, figsize=(45, 35))


    sns.regplot(ax=axs[0, 0], x="danceability", y="popularity", data=df)
    axs[0, 0].set_title('Danceability', fontsize=36)
    print("Plotted Danceability")
    sns.regplot(ax=axs[0, 1], x="energy", y="popularity", data=main_df)
    axs[0, 1].set_title('Energy', fontsize=36)
    print("Plotted Energy")
    sns.regplot(ax=axs[0, 2], x="loudness", y="popularity", data=main_df)
    axs[0, 2].set_title('Loudness', fontsize=36)
    print("Plotted Loudness")
    sns.regplot(ax=axs[0, 3], x="speechiness", y="popularity", data=main_df)
    axs[0, 3].set_title('Speechiness', fontsize=36)
    print("Plotted Speechiness")
    sns.regplot(ax=axs[0, 4], x="acousticness", y="popularity", data=main_df)
    axs[0, 4].set_title('Acousticness', fontsize=36)
    print("Plotted Acousticness")
    sns.regplot(ax=axs[1, 0], x="instrumentalness", y="popularity", data=main_df)
    axs[1, 0].set_title('Instrumentalness', fontsize=36)
    print("Plotted Instrumentalness")
    sns.regplot(ax=axs[1, 1], x="liveness", y="popularity", data=main_df)
    axs[1, 1].set_title('Liveness', fontsize=36)
    print("Plotted Liveness")
    sns.regplot(ax=axs[1, 2], x="valence", y="popularity", data=main_df)
    axs[1, 2].set_title('Valence', fontsize=36)
    print("Plotted Valence")
    sns.regplot(ax=axs[1, 3], x="tempo", y="popularity", data=main_df)
    axs[1, 3].set_title('Tempo', fontsize=36)
    print("Plotted Tempo")
    sns.regplot(ax=axs[1, 4], x="duration_ms", y="popularity", data=main_df)
    axs[1, 4].set_title('Duration', fontsize=36)
    print("Plotted Duration")

    fig.tight_layout(pad=10)
    fig.suptitle("Audio Features Correlated With Current Popularity", fontsize=48)

    #plt.xticks(fontsize=25)

    plt.savefig("feature_popularity_correlation")

def make_word_cloud_cluster(cluster_num):
    scaled_tracks_clustered = pd.read_csv('scaled_tracks_clustered.csv')
    #print(scaled_tracks_clustered)
    tracks_q =  f"""
                            SELECT
                                id
                            FROM
                                scaled_tracks_clustered
                            WHERE
                                cluster = {cluster_num}
                            """
    tracks_df = sqldf(tracks_q, locals()) # because it's not a globally defined table
    track_artists = pd.read_csv("track_artists.csv")
    artist_genres = pd.read_csv("artist_genres.csv")

    genres_q = f"""
                SELECT
                    g.genres
                FROM
                    tracks_df songs
                    INNER JOIN track_artists a on songs.id = a.track_id
                    INNER JOIN artist_genres g on a.artist_id = g.artist_id
    """
    genres_df = sqldf(genres_q, locals()) # because it's not a globally defined table
    print(genres_df)
    genre_text = ' '.join(genres_df['genres'])
    #print(genre_text)
    genre_text_adjusted = genre_text.replace('[','').replace(']','').replace("'", '').replace(",", "")
    #print(genre_text_adjusted)

    from wordcloud import WordCloud, STOPWORDS
    stopwords = STOPWORDS
    stopwords.update(["adult", "standards", "adult standards", "gold", "storm", "album", "road", "mellow", "singer", "songwriter", "easy", "listening"])

    wc = WordCloud(
        background_color='white'
        , stopwords=stopwords
        , height=400
        , width=600
    )

    wc.generate(genre_text_adjusted)
    wc.to_file(f"cluster_{cluster_num}_wordcloud.png")

def main():
    # time_series_plots()

    # get the histograms
    # create_histograms(1958, 2021)
    # create_histograms(1958, 1959)
    # create_histograms(1960, 1969)
    # create_histograms(1970, 1979)
    # create_histograms(1980, 1989)
    # create_histograms(1990, 1999)
    # create_histograms(2000, 2009)
    # create_histograms(2010, 2019)
    # create_histograms(2020, 2021)

    # get the descriptive stats
    # generate_basic_descriptive_statistics(2020, 2021)

    # find_ideal_number_clusters()

    # add_clusters()

    # for i in range(4):
    #     characterize_clusters(i)
    
    # correlate_features_with_curr_popularity() # maybe a bad idea?

    #get_median_popularity(lower_bound=5)
    #append_popular_label()
    
    # popularity_correlations() # maybe these plots are useless?

    # for i in range(4):
    #     make_word_cloud_cluster(cluster_num=i)

    # for i in range(1, 11):
    #     get_important_features_for_popularity_logistic_regression(num_features=i)

    # create_correlation_matrix()
    # aic()
    logistic_regression_popularity()

    pass

if __name__ == "__main__":
    main()
