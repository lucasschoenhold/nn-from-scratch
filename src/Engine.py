import math

from graphviz import Digraph


class Value:
    def __init__(self, data, _children=(), _op: str = None, label: str = ""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward: callable = lambda: None

    def __add__(self, other):
        other = (
            other if isinstance(other, Value) else Value(other)
        )  # Supporting add of integers
        out = Value(self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):  # int + object
        return self.__mul__(other)

    def __pow__(self, other):
        assert isinstance(other, int | float)
        out = Value(self.data**other, (self,), _op=f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other**-1

    def __gt__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self.data > other.data

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad = out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, _children=(self,), _op="tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):  # Needs to be reversed because we start at the end
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data})"


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


# for any value use a rectangle, for any operation use a circle
def draw_graph(value: Value):
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # Left to right
    nodes, edges = trace(value)
    # For each node, add a rectangle with the value
    for n in nodes:
        uid = str(id(n))
        dot.node(
            name=uid,
            label="{%s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
            shape="record",
        )
        # For any operation, use a circle
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            # Add edges to the graph
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot
