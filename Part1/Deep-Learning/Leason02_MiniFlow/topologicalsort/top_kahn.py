# 基于kahn`s algorithm的拓扑排序 与 DFS的拓扑排序有什么不同
from collections import defaultdict, deque

def topological_sort_kahn(graph, num_vertices):
    # 计算每个节点的入度
    in_degree = [0] * num_vertices
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    # 将所有入度为0的节点加入队列
    queue = deque([i for i in range(num_vertices) if in_degree[i] == 0])

    # 拓扑排序结果
    topo_order = []

    while queue:
        u = queue.popleft()
        topo_order.append(u)

        # 遍历邻接节点，减少它们的入度
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # 检查是否有环
    if len(topo_order) == num_vertices:
        return topo_order
    else:
        return []  # 存在环，无法进行拓扑排序


# 示例图
graph = defaultdict(list)
graph[0] = [1, 2]
graph[1] = [3]
graph[2] = [3]
graph[3] = [4]
graph[4] = []

num_vertices = 5

# 执行拓扑排序
result = topological_sort_kahn(graph, num_vertices)
if result:
    print("拓扑排序结果:", result)
else:
    print("图中存在环，无法进行拓扑排序")