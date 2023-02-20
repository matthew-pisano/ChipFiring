from __future__ import annotations
import networkx as nx
import matplotlib.pyplot as plt
from utils import *


class Divisor:
    def __init__(self, chips: list[int]):
        self.rawChips = chips
        # vertical vector for multiplication
        self.divisor = np.array([chips]).transpose()
        self.iterPlace = 0

    def deg(self):
        """Return the degree of the divisor"""
        vSum = 0
        for v in self.divisor:
            vSum += v[0]
        return vSum

    def isEffective(self):
        """If the divisor is winning"""
        for v in self.divisor:
            if v[0] < 0:
                return False
        return True

    def config(self, vertex):
        """Creates a configuration of a divisor by removing an element"""
        return np.delete(self.divisor, vertex, axis=0)

    def copy(self):
        """Creates a copy of this divisor"""
        return Divisor(self.rawChips.copy())

    def __int__(self):
        return self.deg()

    def __add__(self, other):
        self.divisor = np.add(self.divisor, other)

    def __len__(self):
        return len(self.divisor)

    def __getitem__(self, item: int) -> int:
        return self.divisor[item][0]

    def __setitem__(self, key, value):
        self.divisor[key][0] = value

    def __iter__(self):
        self.iterPlace = 0
        return self

    def __next__(self):
        place = self.iterPlace
        if place >= len(self.divisor):
            raise StopIteration()
        self.iterPlace += 1
        return self.divisor[place][0]

    def __str__(self):
        return f"Divisor({np.array(self.divisor.transpose()[0])})"

    def __repr__(self):
        return f"Divisor({np.array(self.divisor.transpose()[0])})"

    def __eq__(self, other):
        for idx, v in enumerate(self.divisor):
            if self[idx] != other[idx]:
                return False
        return True


class Graph:
    """Class representing a graph for chip firing games"""
    BI = 0
    """A bidirectional edge state"""
    FWD = 1
    """A forward edge state"""
    REV = 2
    """A reverse edge state"""

    def __init__(self, adjacency: list[list[int]] | np.ndarray):
        self.matrix = np.copy(adjacency)
        self._undirectedEdges = None
        self._directedEdges = None
        self._refreshState()

    def _refreshState(self):
        """Refresh edge sets and create the laplacian matrix for the graph"""
        self._undirectedEdges = self.edgeSet(directed=False, refresh=True)
        self._directedEdges = self.edgeSet(refresh=True)

        self.laplacian = np.zeros((len(self.matrix),)*2)
        for i in range(0, len(self.laplacian)):
            for j in range(0, len(self.laplacian)):
                self.laplacian[i][j] = self.degree(i) if i == j else -self.matrix[i][j]

    def reducedLaplacian(self, vertex):
        """Creates the reduced laplacian for the given edge"""
        return np.delete(np.delete(self.laplacian, vertex, axis=0), vertex, axis=1)

    def edgeSet(self, vertex=-1, directed=True, refresh=False):
        """Returns either the edge set for a vertex or the edge set for the whole graph"""
        # Return saved calculations for edges by default
        if not refresh and vertex < 0 and not directed:
            return self._undirectedEdges
        if not refresh and vertex < 0 and directed:
            return self._directedEdges

        edgeSet = set()
        if vertex < 0:
            for v in range(0, len(self.matrix)):
                for w in range(0, len(self.matrix)):
                    if (not directed and v > w) or v == w:
                        continue
                    if edge := self.getEdge(v, w, directed):
                        edgeSet.add(edge)
        else:
            for w in range(0, len(self.matrix)):
                if vertex == w:
                    continue
                if edge := self.getEdge(vertex, w, directed):
                    edgeSet.add(edge)
        return edgeSet

    def size(self):
        """Return the number of vertices"""
        return len(self.matrix[0])

    def adjacencySet(self, vertex, directed=True):
        """Return the set of vertices that are adjacent to the given vertex"""
        return [edge[1] for edge in self.edgeSet(vertex, directed)]

    def degree(self, vertex=-1):
        """Returns either the degree of vertex or the degree of the whole graph"""
        if vertex < 0:
            return len(self.matrix)
        else:
            return len(self.adjacencySet(vertex))

    def getEdge(self, v, w, directed=True):
        """Return a directed or undirected edge if it exists"""
        if self.matrix[v][w] > 0 or (not directed and self.matrix[w][v] > 0):
            edgeState = self.BI if self.matrix[v][w] > 0 and self.matrix[w][v] > 0 else \
                self.FWD if self.matrix[v][w] > 0 else self.REV
            return v, w, edgeState

    def auditEdges(self):
        # TODO: rewrite using induction
        sources = 0
        sinks = 0
        for v in range(0, len(self.matrix)):
            edgeSet = list(self.edgeSet(vertex=v, directed=False))
            isSink = True
            isSource = True
            for edge in edgeSet:
                if edge[2] == self.FWD or edge[2] == self.BI:
                    isSink = False
                if edge[2] == self.REV or edge[2] == self.BI:
                    isSource = False
            sources += 1 if isSource else 0
            sinks += 1 if isSink else 0
        return sources, sinks

    def getEdgeState(self, v, w):
        edge = self.getEdge(v, w, directed=False)
        return edge[2] if edge else None

    def getEdgeStates(self):
        """Returns a list of edges and their state"""
        edgeConfig = []
        for i in range(0, len(self.matrix)):
            for j in range(0, len(self.matrix[0])):
                if i > j:
                    continue
                if self.matrix[i][j] != 0 and self.matrix[j][i] != 0:
                    edgeConfig.append((i, j, self.BI))
                elif self.matrix[i][j] != 0:
                    edgeConfig.append((i, j, self.FWD))
                elif self.matrix[j][i] != 0:
                    edgeConfig.append((i, j, self.REV))
        return edgeConfig

    def addEdge(self, v: int, w: int, state: int = 0, weight: int = 1, refreshState=True):
        """Adds an edge between two vertices"""
        if v >= len(self.matrix) and w >= len(self.matrix):
            raise AttributeError("Both vertices cannot be new!")
        elif v >= len(self.matrix) or w >= len(self.matrix):
            self.matrix = np.append(self.matrix, [[0]*len(self.matrix)], 0)
            self.matrix = np.append(self.matrix, [[0]] * (len(self.matrix)), 1)
        self.setEdgeState(v, w, weight=weight, state=state, refreshState=refreshState)

    def setEdgeStates(self, config: list[int]):
        """Set the direction of each edge of the graph using the configuration"""
        edgeSet = list(self.edgeSet(directed=False))
        for idx, state in enumerate(config):
            self.setEdgeState(edgeSet[idx][0], edgeSet[idx][1], state, refreshState=False)
        self._refreshState()

    def setEdgeState(self, v: int, w: int, state: int = 0, weight: int = None, refreshState=True):
        """Sets as an edge as directed or undirected or change the direction of the edge"""
        weight = max(self.matrix[v][w], self.matrix[w][v], 1) if weight is None else weight

        if state == self.BI:
            self.matrix[v][w] = weight
            self.matrix[w][v] = weight
        elif state == self.FWD:
            self.matrix[v][w] = weight
            self.matrix[w][v] = 0
        elif state == self.REV:
            self.matrix[v][w] = 0
            self.matrix[w][v] = weight
        if refreshState:
            self._refreshState()

    def forceFlow(self, v: int, makeSink=True):
        """Force all connections to a vertex to flow in one direction"""
        for edge in self.edgeSet(v, directed=False):
            self.setEdgeState(edge[0], edge[1], 2 if makeSink else 1, refreshState=False)
        self._refreshState()

    def spanningTree(self):
        """Returns the spanning tree for this graph"""
        size = len(self.matrix)
        selectedNode = [0]*size
        noEdge = 0
        selectedNode[0] = True
        tree = self.empty(size)

        while noEdge < size - 1:
            minimum = float('inf')
            a = 0
            b = 0
            for m in range(size):
                if selectedNode[m]:
                    for n in range(size):
                        if not selectedNode[n] and self.matrix[m][n]:
                            # not in selected and there is an edge
                            if minimum > self.matrix[m][n]:
                                minimum = self.matrix[m][n]
                                a = m
                                b = n
            tree.addEdge(a, b, weight=self.matrix[a][b], refreshState=False)
            selectedNode[b] = True
            noEdge += 1
        tree._refreshState()
        return tree

    def lend(self, divisor: Divisor, vertexes: list | np.ndarray | int, amount, forceLegal=False):
        """Lend from a vertex amount number of times. Borrows if amount is negative"""
        if type(vertexes) is int:
            vertexes = [vertexes]

        moves = None
        for vertex in vertexes:
            adj = self.adjacencySet(vertex)
            if len(adj) == 0:
                continue
            # If only legal moves can be made
            if forceLegal:
                adjIllegal = False
                for w in adj:
                    if amount < 0 and divisor[w] + amount < 0:
                        adjIllegal = True
                        break
                # Skip this move if it will cause any vertex to go into an illegal state
                if adjIllegal or (amount > 0 and divisor[vertex] - amount * len(adj)) < 0:
                    continue
            if moves is None:
                moves = [0] * len(divisor)
            divisor[vertex] -= amount * len(adj)
            moves[vertex] -= amount * len(adj)
            for w in adj:
                divisor[w] += amount
                moves[w] += amount

        return moves

    def jac(self, vertex=0, divisor: Divisor = None):
        """Returns an element of Jac(G) that is LinEq to the given divisor"""
        return Utils.coKernel(self.reducedLaplacian(vertex), divisor.config(vertex) if divisor else None)

    def pic(self, divisor: Divisor = None):
        """Returns an element of Pic(G) that is LinEq to the given divisor"""
        return Utils.coKernel(self.laplacian, divisor)

    def visualize(self, divisor: Divisor = None, withWeights=False, title=None, positions=None):
        """Creates a plot representation of the graph"""
        labels = {}
        for i in range(0, len(self.matrix)):
            labels[i] = f"{i}"+(f": {divisor[i]}" if divisor else "")
        G = nx.DiGraph()
        G.add_weighted_edges_from(self.edgeSet())
        pos = nx.circular_layout(G)
        pos = circular_layout(G)
        # pos = nx.planar_layout(G)
        if positions:
            for vertex in positions:
                pos[vertex] = positions[vertex]
        # pos = nx.kamada_kawai_layout(G)
        if withWeights:
            edgeLabels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edgeLabels)
        nx.draw_networkx(G, pos, labels=labels, node_size=700)
        if title:
            plt.title(title)
        plt.show()

    def countPaths(self, cycleRange=None):
        """Counts the number of paths in a cycle graph"""
        if cycleRange is None:
            cycleRange = (0, len(self)-1)
        paths = 0
        firstState = None
        lastState = self.BI
        for i in range(cycleRange[0], cycleRange[1]):
            state = self.getEdgeState(i, i+1)
            if state != lastState and state != self.BI:
                if firstState is None:
                    firstState = state
                paths += 1
                lastState = state
        state = self.getEdgeState(cycleRange[1], cycleRange[0])
        if state != lastState and state != self.BI:
            paths += 1
            lastState = state

        if firstState == lastState and paths != 1:
            paths -= 1

        return paths

    def copy(self):
        """Returns a copy of this graph"""
        return Graph(self.matrix)

    def __len__(self):
        return len(self.matrix)

    @classmethod
    def build(cls, form: str, size: int, **kwargs):
        """Returns a graph of the given form"""
        return getattr(cls, form)(size, **kwargs)

    @classmethod
    def cycle(cls, size, direction=BI):
        """Returns a cycle graph of the given size"""
        graph = cls.empty(size)
        for i in range(0, size-1):
            graph.addEdge(i, i + 1, state=direction, refreshState=False)
        graph.addEdge(size-1, 0, state=direction)
        return graph

    @classmethod
    def wheel(cls, size, axel=0, direction=BI, spokeDirection=BI):
        """Returns a cycle graph of the given size"""
        graph = cls.empty(size)
        rng = list(range(0, size))
        rng.pop(axel)
        for idx, v in enumerate(rng[:-1]):
            graph.addEdge(v, rng[idx + 1], state=direction, refreshState=False)
        graph.addEdge(rng[-1], rng[0], state=direction, refreshState=False)

        for v in rng[:-1]:
            graph.addEdge(axel, v, state=spokeDirection, refreshState=False)
        graph.addEdge(axel, rng[-1], state=spokeDirection)
        return graph

    @classmethod
    def network(cls, shape: list):
        """Returns a network graph with the given shape.  Each element of the shape is
        the number of nodes in that layer of the network"""
        graph = cls.empty(sum(shape))
        prevNodes = 0
        layer = 0
        i = 0
        while i < shape[layer]:
            for j in range(0, shape[layer+1]):
                graph.addEdge(prevNodes+i, prevNodes+shape[layer]+j, state=cls.FWD, refreshState=False)
            i += 1
            if i == shape[layer]:
                prevNodes += shape[layer]
                i = 0
                layer += 1
                if layer == len(shape)-1:
                    break
        graph._refreshState()

        return graph

    @classmethod
    def complete(cls, size):
        """Returns a complete graph of the given size"""
        return Graph(np.ones((size, size))-np.identity(size))

    @classmethod
    def empty(cls, size):
        """Returns an empty graph of the given size"""
        return Graph(np.zeros((size, size)))

    @classmethod
    def _glue(cls, baseGraph: tuple[Graph, int], *subGraphs: tuple[Graph, int], byVertex: bool, edgeState: int = BI):
        baseMatrix = np.copy(baseGraph[0].matrix)
        # Make desired rows and columns last and first for easy connection
        if baseGraph[1] != len(baseMatrix)-1:
            baseMatrix[:, [len(baseMatrix) - 1, baseGraph[1]]] = baseMatrix[:, [baseGraph[1], len(baseMatrix) - 1]]
            baseMatrix[[len(baseMatrix) - 1, baseGraph[1]]] = baseMatrix[[baseGraph[1], len(baseMatrix) - 1]]

        offset = 1 if byVertex else 0

        matrices = []
        for subGraph in subGraphs:
            subMatrix = np.copy(subGraph[0].matrix)
            if subGraph[1] != 0:
                subMatrix[:, [0, subGraph[1]]] = subMatrix[:, [subGraph[1], 0]]
                subMatrix[[0, subGraph[1]]] = subMatrix[[subGraph[1], 0]]
            matrices.append(subMatrix)

        gluedMatrix = baseMatrix
        for matrix in matrices:
            g1Len = len(gluedMatrix)
            glueBase = np.zeros((g1Len+len(matrix)-offset, g1Len+len(matrix)-offset))
            glueBase[0:0 + gluedMatrix.shape[0], 0:0 + gluedMatrix.shape[1]] = gluedMatrix
            glueBase[g1Len-offset:g1Len-offset + matrix.shape[0], g1Len-offset:g1Len-offset + matrix.shape[1]] += matrix

            if not byVertex:
                if edgeState == cls.BI or edgeState == cls.FWD:
                    glueBase[g1Len-1][g1Len] = 1
                if edgeState == cls.BI or edgeState == cls.REV:
                    glueBase[g1Len][g1Len-1] = 1

            gluedMatrix = glueBase

        return Graph(gluedMatrix)

    @classmethod
    def glueByVertex(cls, baseGraph: tuple[Graph, int], *subGraphs: tuple[Graph, int]):
        return cls._glue(baseGraph, *subGraphs, byVertex=True)

    @classmethod
    def glueByEdge(cls, baseGraph: tuple[Graph, int], *subGraphs: tuple[Graph, int], state=0):
        return cls._glue(baseGraph, *subGraphs, byVertex=False, edgeState=state)
