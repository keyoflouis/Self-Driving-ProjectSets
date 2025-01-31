from collections import defaultdict

def topological_sort_dfs(graph, num_vertices):
    visited = [False] * num_vertices
    stack = []

    def dfs(u):
        visited[u] = True
        for v in graph[u]:
            if not visited[v]:
                dfs(v)
        stack.append(u)

    for i in range(num_vertices):
        if not visited[i]:
            dfs(i)

    return stack[::-1]  # 反转栈得到拓扑排序结果

# 示例图
graph = defaultdict(list)
graph[0] = [1, 2]
graph[1] = [3]
graph[2] = [3]
graph[3] = [4]
graph[4] = []

num_vertices = 5

# 执行拓扑排序
result = topological_sort_dfs(graph, num_vertices)
print("拓扑排序结果:", result)