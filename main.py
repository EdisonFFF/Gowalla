import networkx as nx
import pandas as pd


G = nx.read_edgelist(
    "data/loc-gowalla_edges.txt",
    nodetype=int
)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())


print("Density:", nx.density(G))

print("Connected Components:",
      nx.number_connected_components(G))

avg_degree = (
    sum(dict(G.degree()).values())
    / G.number_of_nodes()
)

print("Average Degree:", avg_degree)

checkins = pd.read_csv(
    "data/loc-gowalla_totalCheckins.txt",
    sep="\t",
    header=None,
    names=[
        "user",
        "time",
        "lat",
        "lon",
        "location"
    ]
)

print(checkins.head())

# Degree filtering

degrees = dict(G.degree())

active_degree_users = {
    node for node, degree in degrees.items()
    if degree >= 5
}

print("Users with degree >= 5:",
      len(active_degree_users))


# Check-in filtering

checkin_counts = checkins.groupby("user").size()

active_checkin_users = set(
    checkin_counts[
        checkin_counts >= 10
    ].index
)

print("Users with >=10 check-ins:",
      len(active_checkin_users))

final_users = (
    active_degree_users
    & active_checkin_users
)

print("Final active users:",
      len(final_users))


G_final = G.subgraph(final_users).copy()

print("Filtered Nodes:",
      G_final.number_of_nodes())

print("Filtered Edges:",
      G_final.number_of_edges())