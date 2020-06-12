from math import radians, cos, sin, asin, sqrt

import pandas as pd

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def create_dist_matrix(df):
    df.reset_index(drop=True, inplace=True)

    dist_dict = {"from": [], "to": [], "distance": []}
    dist_matrix = np.zeros([len(df), len(df)])

    for i in df.itertuples():
        idx_i, bs_from, lat_i, lon_i = i[0], i[1], i[2], i[3]

        for j in df.itertuples():
            idx_j, bs_to, lat_j, lon_j = j[0], j[1], j[2], j[3]
            dist_dict["from"].append(bs_from)
            dist_dict["to"].append(bs_to)
            dist_dict["distance"].append(haversine(lon_i, lat_i, lon_j, lat_j))


    return pd.DataFrame.from_dict(dist_dict)


