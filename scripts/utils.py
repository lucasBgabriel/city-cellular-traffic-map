from math import radians, cos, sin, asin, sqrt
import os
import time

from pandas import HDFStore
import pandas as pd
from sklearn.cluster import KMeans

# import clusters

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


def create_distance_df(df):
    """
    :params: df : dataframe with base station id, lat and lon
    """
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


def generate_fogs(df, total_fogs):

    print("generating fogs")
    clusters = []
    centroids = {}

    for i in range(0, total_fogs):
        cluster_i = df[df["cluster"] == i]
        x, c = find_clusters(cluster_i, 3)
        clusters.extend(x)
        centroids[i] = c

    return clusters, centroids

def save_df(df, filename):

    basepath = '/home/giovana/Documentos/personal/city-cellular-traffic-map/data'
    t = time.localtime()
    timestamp = time.strftime("%d-%m_%H-%M", t)

    # filename = f"{name}_{timestamp}"

    df.to_csv(os.path.join(basepath, filename), index=False)
    print(f"dataframe saved in {os.path.join(basepath, filename)}")


def find_clusters(data, k):

    mat = data[['lon', 'lat']]
    kmeans = KMeans(n_clusters = k, random_state = 0).fit(mat)
    return kmeans.labels_, kmeans.cluster_centers_

def load_data():

    basepath = '/home/giovana/Documentos/personal/city-cellular-traffic-map/traceset'
    cellular_traffic_data = pd.read_csv(os.path.join(basepath, 'cellular_traffic.csv'),
            delimiter = ',', decimal='.')
    topology_data = pd.read_csv(os.path.join(basepath, 'topology.csv'), delimiter = ',', decimal='.')

    return cellular_traffic_data, topology_data

def load_clustered_data():
    basepath = '/home/giovana/Documentos/personal/city-cellular-traffic-map/data'

    cellular_traffic_data = pd.read_csv(os.path.join(basepath, 'clustered_traffic.csv'),
            delimiter = ',', decimal='.')
    topology_data = pd.read_csv(os.path.join(basepath, 'clustered_topology.csv'), delimiter = ',', decimal='.')

    return cellular_traffic_data, topology_data

def create_clustered_cellular_df(cellular_traffic_data, topology_data, total_fog, total_rrh, save = False):

    tmp = topology_data[['bs', 'cluster', 'rrh']]
    merged_data = cellular_traffic_data.copy()

    print("merging dataframes")
    merged_data = pd.merge(merged_data, tmp, on='bs', right_index=False, how='left', sort=False)


    cluster_traffic_data = pd.DataFrame(columns=["bs", 'time_hour', 'users', 'packets', 'bytes'])

    for clusters in cellular_traffic_data.groupby(['cluster', 'time_hour']).sum().iterrows():
         cluster_traffic_data = cluster_traffic_data.append({
             'bs': f'fog_{clusters[0][0]}',
             'time_hour': clusters[0][1],
             'users': clusters[1]['users'],
             'packets': clusters[1]['packets'],
             'bytes': clusters[1]['bytes']},
             ignore_index=True)

    for clusters in merged_data.groupby(['cluster', 'rrh', 'time_hour']).sum().iterrows():
         cluster_traffic_data = cluster_traffic_data.append({
             'bs': f'rrh_{clusters[0][0]}_{clusters[0][1]}',
             'time_hour': clusters[0][2],
             'users': clusters[1]['users'],
             'packets': clusters[1]['packets'],
             'bytes': clusters[1]['bytes']}, ignore_index=True)
    if save:
        save_df(merged_data, "cellular_traffic_with_cluster_info.csv")
        save_df(cluster_traffic_data, "clustered_traffic.csv")

    return cluster_traffic_data, merged_data

def create_clustered_topology_df(topology_df, total_fogs, total_rrhs, save=False):

    tmp_topology = topology_df.copy()
    print("finding fogs clusters")
    tmp_topology['cluster'], _ = find_clusters(tmp_topology, total_fogs)
    fog_cluster, fog_centroids = generate_fogs(tmp_topology, total_fogs)

    # marking which rrh belongs to each fog node
    tmp_topology['rrh'] = fog_cluster


    clustered_topology = pd.DataFrame(columns=["bs", "lat", "lon"])
    for clusters in tmp_topology.groupby(['cluster']).mean().iterrows():
        clustered_topology = clustered_topology.append(
                {'bs': f'fog_{clusters[0]}',
                'lat': clusters[1]['lat'],
                'lon': clusters[1]['lon']}, ignore_index=True)

    for clusters in tmp_topology.groupby(['cluster', 'rrh']).mean().iterrows():
         clustered_topology = clustered_topology.append(
                 {'bs': f'rrh_{clusters[0][0]}_{clusters[0][1]}',
                     'lat': clusters[1]['lat'],
                     'lon': clusters[1]['lon']}, ignore_index=True)

    if save:
        save_df(tmp_topology, "topology_with_clusters_and_rrhs_info.csv")
        save_df(clustered_topology, "clustered_topology.csv")

    return clustered_topology, tmp_topology

def correct_time_field(df):

    correct_df = df.copy()
    correct_df['time_hour'] = correct_df['time_hour'].map(
            lambda x: pd.Timestamp(x, unit='s', tz='Asia/Shanghai'))
    correct_df['hour'] = correct_df['time_hour'].map(lambda x: x.hour)
    correct_df['day'] = correct_df['time_hour'].map(lambda x: x.day)

    return df

def generate_hd5_file(pivot_df, name):

    store = HDFStore(f"data/{name}.h5")
    store.put(name, pivot_df, format='table', data_columns=True)
    store.close()



