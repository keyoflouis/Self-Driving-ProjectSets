"""
This script builds and runs a graph with miniflow.

There is no need to change anything to solve this quiz!

However, feel free to play with the network! Can you also
build a network that solves the equation below?

(x + y) + y
"""

from miniflow import *

x, y = Input(), Input()

f = Add(x, y)

feed_dict = {x: 10, y: 5}

# 使用拓扑排序的同时输入值
sorted_nodes = topological_sort(feed_dict)

# 对每一个节点使用前向传播的方法（Node基类提供的接口），Input 和 Add节点实现不同
output = forward_pass(f, sorted_nodes)

# NOTE: because topological_sort sets the values for the `Input` nodes we could also access
# the value for x with x.value (same goes for y).
print("{} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], output))