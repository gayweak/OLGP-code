import time
import pymysql
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import random


start=time.perf_counter()
connection=pymysql.connect(
    host='127.0.0.1',
    port=3306,
    user='root',
    password='098831',
    database='database1'
)
cursor=connection.cursor()

sql="select StartID,EndID from Relations where time=2022"
cursor.execute(sql)
relations=cursor.fetchall()
relations=[item[0] for item in relations]

def Breadth_First_Search(C, F, source, sink):
    """寻找增广路径，并返回路径与剩余容量"""
    queue = deque([source])
    paths = {source: []}
    if source == sink:
        return paths[source]
    while queue:
        u = queue.popleft()
        for v in C:
            if C[u][v] - F[u][v] > 0 and v not in paths:
                paths[v] = paths[u] + [(u, v)]
                if v == sink:
                    return paths[v]
                queue.append(v)
    return None


def maximal_flow(C, source, sink):
    """Ford-Fulkerson 算法，计算从 source 到 sink 的最大流"""
    n = len(C)
    F = [[0] * n for _ in range(n)]
    path = Breadth_First_Search(C, F, source, sink)
    while path is not None:
        flow = min(C[u][v] - F[u][v] for u, v in path)
        for u, v in path:
            F[u][v] += flow
            F[v][u] -= flow
        path = Breadth_First_Search(C, F, source, sink)
    return sum(F[source][i] for i in range(n))


def weights_computing(G1):
    """计算图 G1 中每条边的最大流并更新其权重"""
    for u, v in G1.edges():
        # 创建容量矩阵 C
        nodes = list(G1.nodes())
        n = len(nodes)
        C = [[0] * n for _ in range(n)]
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        for edge_u, edge_v in G1.edges():
            idx_u = node_to_idx[edge_u]
            idx_v = node_to_idx[edge_v]
            C[idx_u][idx_v] = 1  # 初始容量为 1

        # 使用 Ford-Fulkerson 算法计算最大流
        max_flow = maximal_flow(C, node_to_idx[u], node_to_idx[v])

        # 更新边的权重为最大流
        G1[u][v]['weight'] = max_flow

def graph(G):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_color='black',
            edge_color='gray')
    plt.title("visualization of Graph G2")
    plt.show()

def maximal_clique_mining(G):
    def extend_clique(potential_clique, remaining_nodes, skip_nodes, cliques):
        if not remaining_nodes and not skip_nodes:
            if len(potential_clique) > 3:
                cliques.append(potential_clique)
            return

        # 选择一个枢轴节点
        pivot = max(remaining_nodes | skip_nodes, key=lambda node: G.degree(node), default=None)
        neighbors_of_pivot = set(G.neighbors(pivot))

        # 优先处理不与枢轴相邻的节点
        for node in remaining_nodes - neighbors_of_pivot:
            new_potential_clique = potential_clique | {node}
            new_remaining_nodes = remaining_nodes & set(G.neighbors(node))
            new_skip_nodes = skip_nodes & set(G.neighbors(node))
            extend_clique(new_potential_clique, new_remaining_nodes, new_skip_nodes, cliques)
            remaining_nodes.remove(node)
            skip_nodes.add(node)

    maximal_cliques = []
    # 按照度排序剩余节点
    initial_remaining = sorted(G.nodes(), key=lambda node: G.degree(node), reverse=True)
    extend_clique(set(), set(initial_remaining), set(), maximal_cliques)
    return maximal_cliques

def pruning_edges(G):
    degrees = [G.degree(node) for node in G.nodes()]
    threshold = np.percentile(degrees, 15)
    nodes_to_remove = [node for node in G.nodes() if G.degree(node) < threshold]
    G.remove_nodes_from(nodes_to_remove)

def k_means_clustering(G, k, max_iters=100):
    nodes = list(G.nodes())
    adjacency_matrix = [[0] * len(nodes) for _ in range(len(nodes))]

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if nodes[j] in G[nodes[i]]:
                adjacency_matrix[i][j] = 1

    n = len(adjacency_matrix)
    centroids = random.sample(nodes, k)  # 随机选择初始中心
    print(f"Initial centroids: {centroids}")

    for iteration in range(max_iters):
        clusters = {i: [] for i in range(k)}
        print(f"\nIteration {iteration + 1}:")

        # 分配节点到最近的质心
        for i in range(n):
            distances = [euclidean_distance(adjacency_matrix[i], adjacency_matrix[nodes.index(centroid)]) for centroid
                         in centroids]
            cluster_index = distances.index(min(distances))
            clusters[cluster_index].append(nodes[i])

        # 计算新的质心
        new_centroids = {}
        for i in range(k):
            if clusters[i]:
                new_centroids[i] = calculate_centroid(clusters[i], adjacency_matrix)
            else:
                # 如果某个簇为空，随机选择一个节点作为新的质心
                new_centroids[i] = random.choice(nodes)

        print(f"Clusters: {clusters}")
        print(f"New centroids: {new_centroids}")

        if new_centroids == centroids:
            print("Convergence reached.")
            break
        centroids = new_centroids

    return clusters

def euclidean_distance(vec1, vec2):
    return sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5

def calculate_centroid(cluster, adjacency_matrix):
    n = len(cluster)
    centroid = [0] * len(adjacency_matrix[0])
    for node in cluster:
        centroid = [centroid[i] + adjacency_matrix[node.index(node)][i] for i in range(len(centroid))]
    return [x / n for x in centroid]


def CGC(G,G1,maximal_cliques):
 clique_to_node={}
 for i,clique in enumerate(maximal_cliques):
    new_node=f"Clique{i+1}"
    G1.add_node(new_node)
    clique_to_node[frozenset(clique)]=new_node

 for clique1,node1 in clique_to_node.items():
    for clique2,node2 in clique_to_node.items():
        if node1 != node2:
            connected=False
            for node_in_clique1 in clique1:
                for node_in_clique2 in clique2:
                    if G.has_edge(node_in_clique1,node_in_clique2):
                        connected=True
                        break
                if connected:
                    break
            if connected:
                G1.add_edge(node1,node2)


G=nx.Graph()
for start_id,end_id in relations:
    G.add_edge(start_id,end_id)

maximal_cliques=maximal_clique_mining(G)

G1=nx.Graph()
clusters1 = k_means_clustering(G1,k=2)
CGC(G,G1,maximal_cliques)
pruning_edges(G1)
weights_computing(G1)
for start_id, end_id, data in G1.edges(data=True):
    weight = data['weight']  # 直接获取边的权重
    sql = "INSERT INTO Relation(StartID, EndID, Weight, time) VALUES (%s, %s, %s, %s)"
    cursor.execute(sql, (start_id, end_id, weight, 2022))
graph(G1)

maximal_cliques=maximal_clique_mining(G1)
G2=nx.Graph()
CGC(G1,G2,maximal_cliques)
clusters2 = k_means_clustering(G2,k=3)
pruning_edges(G2)
weights_computing(G2)
for start_id, end_id, data in G2.edges(data=True):
    weight = data['weight']  # 直接获取边的权重
    sql = "INSERT INTO Relation(StartID, EndID, Weight, time) VALUES (%s, %s, %s, %s)"
    cursor.execute(sql, (start_id, end_id, weight, 2022))
graph(G2)
maximal_cliques=maximal_clique_mining(G2)

G3=nx.Graph()
CGC(G2,G3,maximal_cliques)
clusters3 = k_means_clustering(G3,k=4)
pruning_edges(G3)
weights_computing(G3)
for start_id, end_id, data in G3.edges(data=True):
    weight = data['weight']  # 直接获取边的权重
    sql = "INSERT INTO Relation(StartID, EndID, Weight, time) VALUES (%s, %s, %s, %s)"
    cursor.execute(sql, (start_id, end_id, weight, 2022))
graph(G3)


connection.commit()
cursor.close()
connection.close()
end=time.perf_counter()
print(end-start)
