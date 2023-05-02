from dotenv import load_dotenv
import os
import base64
from requests import post, get
import json
import csv
import time
import itertools
import sys


load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

def get_token():
    auth_string = client_id + ":" + client_secret
    #print(auth_string)
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "client_credentials"
    }
    result = post(url, headers=headers, data=data)
    #print(result.status_code)
    json_result = json.loads(result.content)
    #print(json_result)
    token = json_result["access_token"]
    return token

def get_auth_header(token):
    return {"Authorization": "Bearer " + token}

def search_for_artist(token, artist_name):
    url = "https://api.spotify.com/v1/search"
    headers = get_auth_header(token)
    query = f"?q={artist_name}&type=artist&limit=1"

    query_url = url + query
    result = get(query_url, headers=headers)
    json_result = json.loads(result.content)["artists"]["items"]
    if len(json_result) == 0:
        print("No artist with this name exists...")
        return None

    return json_result[0]

def search_for_track(token, track_name, artist):
    url = "https://api.spotify.com/v1/search"
    headers = get_auth_header(token)
    #query = f"?q=name={track_name}year={year}artist={artist}&type=track&limit=1"

    query = f"?q={track_name}%20{artist}%20&type=track&limit=1"

    query_url = url + query
    result = get(query_url, headers=headers)
    # print("Whole Result: ", result)
    # print("Result Headers: ", result.headers)

    if result.status_code == 401:   # probably need to refresh token?
        print(f"Ran into a {result.status_code}, with track = {track_name} and artist = {artist}.")
        print("Whole Result: ", result)
        print("Result Headers: ", result.headers)
        return 401

    if result.status_code != 200 and result.status_code != 429:
        print(f"Ran into a {result.status_code}, with track = {track_name} and artist = {artist}.")
        time.sleep(10)
        return None # just don't feel like handling this, we'll still get a lot of data anyway

    wait_adder = 2
    while result.status_code == 429:
        print("Ran into a 429... need to wait...")
        time_to_wait = int(result.headers["Retry-After"])
        print("time_to_wait = ", time_to_wait)
        print("wait_adder = ", wait_adder)
        total_wait_time = time_to_wait + wait_adder
        print("total_wait_time = ", total_wait_time)
        time.sleep(total_wait_time)    # Try waiting extra time if we sent too many
        wait_adder += 10                         # Wait longer if we need to
        result = get(query_url, headers=headers)

    json_result = json.loads(result.content)['tracks']['items']
    # print(json_result)
    if len(json_result) == 0:
        print("No track with this name exists...")
        return None

    return json_result[0]

def get_track_info(token):
    with open('distinct_track_ids.csv', 'r') as track_list:
        with open('track_features_1.csv', 'w') as features_file:
            values = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'type', 'id', 'uri', 'track_href', 'analysis_url', 'duration_ms', 'time_signature']
            datareader = csv.reader(track_list)
            writer = csv.DictWriter(features_file, fieldnames=values)
            writer.writeheader()
            next(datareader)
            group_number = 0
            while True:
                next_100_tracks = [x.rstrip('\n') for x in itertools.islice(track_list, 100)]
                if not next_100_tracks:
                    break

                # process next_n_lines
                first_track = next_100_tracks[0]
                last_track = next_100_tracks[-1]
                print(f"Retrieving group {group_number} ; tracks {first_track} through {last_track}")
                time.sleep(0.5) # just trying to avoid a 429

                track_query_string = ",".join(map(str, next_100_tracks))
                url = "https://api.spotify.com/v1/audio-features"
                headers = get_auth_header(token)
                #query = f"?q=name={track_name}year={year}artist={artist}&type=track&limit=1"

                query = f"?ids={track_query_string}"
                print("query: ", query)

                query_url = url + query
                result = get(query_url, headers=headers)

                json_result = json.loads(result.content)
                # print(json_result)
                feature_set = json_result["audio_features"]
                for track_features in feature_set:
                    if track_features == None: 
                        continue
                    else:                        
                        print(track_features)
                        writer.writerow(track_features)

                group_number += 1   # should have like 285 groups?


def get_track_popularity(token):
    with open('distinct_track_ids.csv', 'r') as track_list:
        with open('track_popularity.csv', 'w') as popularity_file:
            values = ['id', 'popularity']
            datareader = csv.reader(track_list)
            writer = csv.DictWriter(popularity_file, fieldnames=values)
            writer.writeheader()
            next(datareader)
            group_number = 0
            while True:
                next_100_tracks = [x.rstrip('\n') for x in itertools.islice(track_list, 50)]
                if not next_100_tracks:
                    break

                # process next_n_lines
                first_track = next_100_tracks[0]
                last_track = next_100_tracks[-1]
                print(f"Retrieving group {group_number} ; tracks {first_track} through {last_track}")
                time.sleep(0.5) # just trying to avoid a 429

                track_query_string = ",".join(map(str, next_100_tracks))
                url = "https://api.spotify.com/v1/tracks"
                headers = get_auth_header(token)
                #query = f"?q=name={track_name}year={year}artist={artist}&type=track&limit=1"

                query = f"?ids={track_query_string}"
                #print("query: ", query)

                query_url = url + query
                result = get(query_url, headers=headers)
                json_result = json.loads(result.content)
                #print(json_result)
                tracks = json_result["tracks"]
                for track in tracks:
                    if track == None: 
                        continue
                    else:                        
                        #print(track["popularity"])
                        popularity = track["popularity"]
                        id = track["id"]
                        print(f"id = {id}; popularity = {popularity}")
                        writer.writerow({
                                            "id": id, 
                                            "popularity": popularity
                                         })

                group_number += 1   # should have like 285 groups?

def get_artists(token):
    with open('distinct_track_ids.csv', 'r') as track_list:
        with open('track_artists.csv', 'w') as write_file:
            values = ['track_id', 'artist_id']
            datareader = csv.reader(track_list)
            writer = csv.DictWriter(write_file, fieldnames=values)
            writer.writeheader()
            next(datareader)
            group_number = 0
            while True:
                next_100_tracks = [x.rstrip('\n') for x in itertools.islice(track_list, 50)]
                if not next_100_tracks:
                    break

                # process next_n_lines
                first_track = next_100_tracks[0]
                last_track = next_100_tracks[-1]
                print(f"Retrieving group {group_number} ; tracks {first_track} through {last_track}")
                time.sleep(0.5) # just trying to avoid a 429

                track_query_string = ",".join(map(str, next_100_tracks))
                url = "https://api.spotify.com/v1/tracks"
                headers = get_auth_header(token)
                #query = f"?q=name={track_name}year={year}artist={artist}&type=track&limit=1"

                query = f"?ids={track_query_string}"
                #print("query: ", query)

                query_url = url + query
                result = get(query_url, headers=headers)
                json_result = json.loads(result.content)
                #print(json_result)
                tracks = json_result["tracks"]
                for track in tracks:
                    if track == None: 
                        continue
                    else:                        
                        #print(track["popularity"])
                        artists = track["artists"]
                        id = track["id"]
                        for artist in artists:
                            artist_id = artist["id"]
                            print(f"id = {id}; artist = {artist_id}")
                            writer.writerow({
                                                "track_id": id, 
                                                "artist_id": artist_id
                                             })

                group_number += 1   # should have about 600 groups?



def get_artist_genres(token):
    with open('distinct_artists.csv', 'r') as artist_list:
        with open('artist_genres.csv', 'w') as write_file:
            values = ['artist_id', 'genres']
            datareader = csv.reader(artist_list)
            writer = csv.DictWriter(write_file, fieldnames=values)
            writer.writeheader()
            next(datareader)
            group_number = 0
            while True:
                next_100_tracks = [x.rstrip('\n') for x in itertools.islice(artist_list, 50)]
                if not next_100_tracks:
                    break

                # process next_n_lines
                first_track = next_100_tracks[0]
                last_track = next_100_tracks[-1]
                print(f"Retrieving group {group_number} ; artists {first_track} through {last_track}")
                time.sleep(0.5) # just trying to avoid a 429

                track_query_string = ",".join(map(str, next_100_tracks))
                url = "https://api.spotify.com/v1/artists"
                headers = get_auth_header(token)
                #query = f"?q=name={track_name}year={year}artist={artist}&type=track&limit=1"

                query = f"?ids={track_query_string}"
                #print("query: ", query)

                query_url = url + query
                result = get(query_url, headers=headers)
                json_result = json.loads(result.content)
                # print(json_result)
                artists = json_result["artists"]
                for artist in artists:
                    #print(artist)
                    if artist == None: 
                        continue
                    else:                        
                        #print(track["popularity"])
                        genres = artist["genres"]
                        artist_id = artist["id"]
                        print(f"artist_id = {artist_id}; genres = {genres}")
                        writer.writerow({
                                            "artist_id": artist_id, 
                                            "genres": genres
                                            })

                group_number += 1   # should have about 600 groups?

def get_track_ids(token):
    
    # don't need to retrieve songs more than once
    retrieved_songs = {}

    # open the charts.csv so we can gather the song ids:
    filename = './charts.csv'

    with open(filename, 'r') as csvfile:
        with open("new_tracks_with_spotify_ids_3.csv", "w") as new_csvfile: #TODO
            column_headers = ["given_date", "given_rank", "track_id", "spotify_artist", "given_artist", "spotify_title", "given_title", "given_last_week", "given_peak_rank", "given_weeks_on_board"]
            filewriter = csv.DictWriter(new_csvfile, fieldnames=column_headers)
            filewriter.writeheader()
            datareader = csv.reader(csvfile)
            #next(datareader)    # skip header row
            # counter = 0
            for row in itertools.islice(datareader, 233048, None): #TODO
                # counter += 1

                # # testing the waters so I don't overuse the API
                # if counter >= 10: 
                #     break
                
                given_date = row[0]
                given_rank = row[1]
                given_title = row[2]
                given_artist = row[3]
                given_last_week = row[4]
                given_peak_rank = row[5]
                given_weeks_on_board = row[6]
                song_tuple = (given_title, given_artist)
                if song_tuple in retrieved_songs:
                    (track_id, spotify_artist, spotify_title) = retrieved_songs.get(song_tuple)
                    filewriter.writerow({
                                            "given_date": given_date, 
                                            "given_rank": given_rank,
                                            "track_id": track_id, 
                                            "spotify_artist": spotify_artist, 
                                            "given_artist": given_artist, 
                                            "spotify_title": spotify_title, 
                                            "given_title": given_title,
                                            "given_last_week": given_last_week, 
                                            "given_peak_rank": given_peak_rank, 
                                            "given_weeks_on_board": given_weeks_on_board
                                         })
                else:
                    time.sleep(0.5)   # apparently don't exceed 0.33 request per second, per rate limit note
                    search_title = given_title.replace('#', '')
                    search_artist = given_artist.replace('#', '')
                    result = search_for_track(token, search_title, search_artist)

                    if result == None: 
                        continue

                    if result == 401:
                        token = get_token() # it could just be that our token's expired (after 1 hour), so try getting another?
                        time.sleep(10)  # just to be safe?
                        result = search_for_track(token, search_title, search_artist)

                    while result == "status_code_429":
                        print("Maybe sent too many requests, try waiting...")
                        time.sleep(10)
                    track_id = result["id"]
                    spotify_title = result["name"]
                    spotify_artist = result["artists"][0]["name"]
                    filewriter.writerow({
                                            "given_date": given_date, 
                                            "given_rank": given_rank,
                                            "track_id": track_id, 
                                            "spotify_artist": spotify_artist, 
                                            "given_artist": given_artist, 
                                            "spotify_title": spotify_title, 
                                            "given_title": given_title,
                                            "given_last_week": given_last_week, 
                                            "given_peak_rank": given_peak_rank, 
                                            "given_weeks_on_board": given_weeks_on_board
                                         })
                    retrieved_songs[(given_title, given_artist)] = (track_id, spotify_artist, spotify_title)
                print(row)

def main():
    token = get_token()
    # result = search_for_track(token, "Hot Line", "The Sylvers") # used to take a year as input
    # track_id = result["id"]
    # title = result["name"]
    # artist = result["artists"][0]["name"]
    # print("track id: ", track_id, "; title: ", title, "; artist: ", artist)

    #get_track_ids(token)
    #get_track_info(token)
    # get_track_popularity(token)
    # get_artists(token)
    get_artist_genres(token)

if __name__ == "__main__":
    main()