import pandas as pd
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

# data = pd.read_csv("demo.csv")
# data = data.set_index("id")

# print(data.query("age >= 30"))

# sql_query = """
#             SELECT * FROM data
#             """

# filtered = ps.sqldf(sql_query, locals())
# print(filtered)

data = pd.read_csv("track_artists.csv")

print(data)

sql_query = """
            SELECT DISTINCT artist_id FROM data
            """
filtered = pysqldf(sql_query)
print(filtered)
filtered.to_csv('distinct_artists.csv', encoding='utf-8', index=False)