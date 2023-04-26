import pandas as pd
import pandasql as ps

# data = pd.read_csv("demo.csv")
# data = data.set_index("id")

# print(data.query("age >= 30"))

# sql_query = """
#             SELECT * FROM data
#             """

# filtered = ps.sqldf(sql_query, locals())
# print(filtered)

data = pd.read_csv("new_tracks_with_spotify_ids_2.csv")

print(data)

sql_query = """
            SELECT DISTINCT track_id FROM data
            """
filtered = ps.sqldf(sql_query, locals())
print(filtered)