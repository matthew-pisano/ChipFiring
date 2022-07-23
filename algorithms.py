import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Divisor:
    def __init__(self, chips: list[int]):
        self.rawChips = chips
        # vertical vector for multiplication
        self.divisor = np.array(chips).transpose()
        self.iterPlace = 0

    def deg(self):
        """Return the degree of the divisor"""
        vsum = 0
        for v in self.divisor:
            vsum += v
        return vsum

    def isEffective(self):
        """If the divisor is winning"""
        for v in self.divisor:
            if v < 0:
                return False
        return True

    def copy(self):
        return Divisor(self.rawChips.copy())

    def __int__(self):
        return self.deg()

    def __len__(self):
        return len(self.divisor)

    def __getitem__(self, item: int) -> int:
        return self.divisor[item]

    def __setitem__(self, key, value):
        self.divisor[key] = value

    def __iter__(self):
        self.iterPlace = 0
        return self

    def __next__(self):
        place = self.iterPlace
        if place >= len(self.divisor):
            raise StopIteration()
        self.iterPlace += 1
        return self.divisor[place]

    def __str__(self):
        return f"Divisor({self.divisor})"

    def __repr__(self):
        return f"Divisor({self.divisor})"


class Graph:
    def __init__(self, adjacency):
        self.matrix = adjacency
        self.laplacian = self._makeLaplacian()

    def _makeLaplacian(self):
        """Create the laplacian matrix for the graph"""
        laplacian = np.zeros((len(self.matrix),)*2)
        for i in range(0, len(laplacian)):
            for j in range(0, len(laplacian)):
                laplacian[i][j] = self.degree(i) if i == j else -self.commonEdges(i, j)
        return laplacian

    def edgeSet(self, vertex=-1):
        """Returns either the edge set for a vertex or the edge set for the whole graph"""
        if vertex < 0:
            return [(v, w) for w in range(0, len(self.matrix)) for v in range(0, len(self.matrix))
                    for _ in range(0, self.matrix[v][w]) if v < w]

        return [(vertex, w) for w in range(0, len(self.matrix))
                for _ in range(0, self.matrix[vertex][w]) if vertex != w]

    def commonEdges(self, v, w):
        return self.matrix[v][w]

    def size(self):
        """Return the number of vertices"""
        return len(self.matrix[0])

    def adjacencySet(self, vertex):
        """Return the set of vertices that are adjacent to the given vertex"""
        return [edge[1] for edge in self.edgeSet(vertex)]

    def degree(self, vertex=-1):
        """Returns either the degree of vertex or the degree of the whole graph"""
        if vertex < 0:
            return len(self.matrix)
        else:
            return len(self.adjacencySet(vertex))

    def hasEdge(self, v, w):
        """Return whether two vertices share an edge"""
        edges = self.edgeSet(v)
        return (v, w) in edges or (w, v) in edges

    def lend(self, divisor: Divisor, vertexes: list | int, amount):
        """Lend from a vertex amount number of times.
        Borrows if amount is negative"""
        if type(vertexes) is int:
            vertexes = [vertexes]

        moves = [0] * len(divisor)
        for vertex in vertexes:
            adj = self.adjacencySet(vertex)
            divisor[vertex] -= amount * len(adj)
            moves[vertex] -= amount * len(adj)
            for w in adj:
                divisor[w] += amount
                moves[w] += amount

        return moves

    def visualize(self, divisor: Divisor):
        """Creates a plot representation of the grapg"""
        labels = {}
        for i in range(0, len(divisor)):
            labels[i] = f"{i}: {divisor[i]}"
        G = nx.Graph()
        G.add_edges_from(self.edgeSet())
        nx.draw_networkx(G, labels=labels, node_size=700)
        plt.show()


def greedy(graph: Graph, divisor: Divisor, mutate=False):
    """Finds an equivalent, effective divisor to the one given using
    the greedy algorithm"""
    if divisor.deg() < 0:
        return False
    marked = []
    moves = []
    if not mutate:
        divisor = divisor.copy()

    while not divisor.isEffective():
        if len(marked) < len(divisor):
            for i in range(0, len(divisor)):
                if divisor[i] < 0:
                    moves.append(graph.lend(divisor, i, -1))
                    if i not in marked:
                        marked.append(i)
                    break
        else:
            return False, None, None
    return True, moves, divisor


def dhar(graph: Graph, divisor: Divisor, source: int, burned: list[int] = None):
    """Determines if a divisor has a super-stable configuration"""
    if burned is None:
        burned = [source]

    for v in graph.adjacencySet(source):
        if v in burned:
            continue
        threats = 0
        for b in burned:
            if graph.hasEdge(v, b):
                threats += 1
        if divisor[v] < threats:
            burned.append(v)
            dhar(graph, divisor, v, burned)

    return len(burned) == len(divisor)


def qReduced(graph: Graph, divisor: Divisor, mutate=False):
    q = 0
    if not mutate:
        divisor = divisor.copy()
    moves = []

    def needMore():
        for i, v in enumerate(divisor):
            if v < 0 and i != q:
                return True
        return False
    # Stage 1: Benevolence
    while needMore():
        moves.append(graph.lend(divisor, q, 1))
        for i, v in enumerate(divisor):
            # Lend only lend if vertex will not go into debt
            if v >= len(graph.adjacencySet(i)) and i != q:
                moves.append(graph.lend(divisor, i, 1))

    # Stage 2: Relief
    legals = [i for i in range(0, len(divisor)) if i != q]
    while divisor[q] < 0 and len(legals) > 0:
        lIdx = 0
        while lIdx < len(legals):
            adjSet = graph.adjacencySet(legals[lIdx])
            count = 0
            for i in adjSet:
                if i not in legals:
                    count += 1
            if count > divisor[legals[lIdx]]:
                legals.pop(lIdx)
                lIdx = 0
            else:
                lIdx += 1

        graph.lend(divisor, legals, 1)

    if divisor[q] < 0:
        return False, None, None
    return True, moves, divisor

