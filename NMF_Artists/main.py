import pandas as pd
import numpy as np
import csv
import warnings
from sklearn.decomposition import NMF
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from scipy.sparse import csr_matrix


def matrix_setup():
    df = pd.read_csv('scrobbler-small-sample.csv', usecols=[0, 1, 2])
    new_df = pd.pivot_table(df, index='artist_offset', columns='user_offset', values='playcount')
    new_df_dropna = new_df.replace(np.nan, 0)
    new_df_dropna.drop_duplicates(inplace=True)
    sparse_matrix = csr_matrix(new_df_dropna)
    return sparse_matrix


def parse_input(names: list):
    print("Enter the names of some of your favourite artists (MAX 3):")
    user_input = input().split(", ")

    if len(user_input) > 3:
        return False
    for thing in user_input:
        if thing not in names:
            print("ERROR: Artist {x} not found".format(x=thing))
            return False
    return user_input


def multiple_similar_artist(artists, df):
    main_df = similar_artist(df.loc[artists[0]], df)
    temp_artists = artists[1:]
    for name in temp_artists:
        temp = df.loc[name]
        temp_df = similar_artist(temp, df)
        main_df = main_df.merge(temp_df, left_index=True, right_index=True)
    return main_df

def similar_artist(artist, df):
    return pd.DataFrame(df.dot(artist).nlargest(10))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    artists = matrix_setup()

    artist_names = []
    with open('artists.csv', newline='') as f:
        for row in csv.reader(f):
            artist_names.append(row[0])

    scalar = MaxAbsScaler()
    nmf = NMF(n_components=20)
    normalizer = Normalizer()
    pipeline = make_pipeline(scalar, nmf, normalizer)
    norm_features = pipeline.fit_transform(artists)

    df = pd.DataFrame(norm_features, index=artist_names)

    temp = parse_input(artist_names)
    while temp is False:
        temp = parse_input(artist_names)

    recommends = multiple_similar_artist(temp, df)
    for name in temp:
        if name in recommends.index.values:
            recommends.drop(name, inplace=True)

    print("Based on input, the following artists are recommended:")
    print(recommends.index.values)

    # print(similar_artist(df.loc[temp[0]], df))
    # print(similar_artist(df.loc[temp[1]], df))