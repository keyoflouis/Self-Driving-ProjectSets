"""
Write the Linear#forward method below!
"""


class Node:
    def __init__(self, inbound_nodes=[]):
        # Nodes from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Nodes to which this Node passes values
        self.outbound_nodes = []
        # A calculated value
        self.value = None
        # Add this node as an outbound node on its inputs.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

    # These will be implemented in a subclass.
    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_nodes` and
        store the result in self.value.
        """
        raise NotImplemented


class Input(Node):
    def __init__(self):
        # An Input Node has no inbound nodes,
        # so no need to pass anything to the Node instantiator
        Node.__init__(self)

        # NOTE: Input Node is the only Node where the value
        # may be passed as an argument to forward().
        #
        # All other Node implementations should get the value
        # of the previous nodes from self.inbound_nodes
        #
        # Example:
        # val0 = self.inbound_nodes[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value is not None:
            self.value = value


class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

        # NOTE: The weights and bias properties here are not
        # numbers, but rather references to other nodes.
        # The weight and bias values are stored within the
        # respective nodes.

    def forward(self):
        """
        Set self.value to the value of the linear function output.

        Your code goes here!
        """

        inputs = self.inbound_nodes[0].value
        weights =self.inbound_nodes[1].value
        bias = self.inbound_nodes[2].value

        self.value = sum(i*w for i ,w in zip(inputs,weights)) +bias




def topological_sort(feed_dict):

    """
    Sort the nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Node and the value is the respective value feed to that Node.

    Returns a list of sorted nodes.
    """

    # 将键 [input,weight,bias] 赋值给input_nodes,这里的键是Input对象
    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    # 复制 input_nodes 给nodes
    nodes = [n for n in input_nodes]


    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
            # G的结构
            # G = {
            #    A: {'in': set(), 'out': {B, C}},
            #    B: {'in': {A}, 'out': {D}},
            #    C: {'in': {A}, 'out': {D}},
            #    D: {'in': {B, C}, 'out': set()}
            # }
            #

        # 根据节点的 outbound_nodes 属性,设置G
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    # 这一段的 input_nodes 本就是从 feed_dict 获得的key,
    # inode_nodes 是一个 Input 类 (Input类对象在初始时没有设置value,我们需要把它对应的列表赋值给他)
    # 检查它是否为 Input 类型,并且重新将feed_dict的值复制给这个n
    while len(S) > 0:
        n = S.pop()
        if isinstance(n, Input):
            n.value = feed_dict[n]
        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.

    Returns the output Node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value