import pandas as pd

df = pd.concat(map(pd.read_csv, ['new_tracks_with_spotify_ids_1.csv', 'new_tracks_with_spotify_ids_2.csv', 'new_tracks_with_spotify_ids_3.csv']), ignore_index=True)

#print(df)

df.to_csv('final_track_ids.csv', encoding='utf-8', index=False)

# Creating the distinct list of track ids
# >>> import pandas as pd
# >>> df = pd.read_csv('charts_with_spotify_ids_final.csv')
# >>> import pandasql as ps
# >>> sql_query = "SELECT DISTINCT track_id FROM df"
# >>> filtered = ps.sqldf(sql_query, locals())
# >>> filtered.to_csv('distinct_track_ids.csv', encoding='utf-8', index=False)
# >>> exit()