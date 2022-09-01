from __future__ import annotations
import random
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils import logger, Utils


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
                    if self.matrix[v][w] > 0 or (not directed and self.matrix[w][v] > 0):
                        edgeState = self.BI if self.matrix[v][w] > 0 and self.matrix[w][v] > 0 else \
                            self.FWD if self.matrix[v][w] > 0 else self.REV
                        edgeSet.add((v, w, edgeState))
        else:
            for w in range(0, len(self.matrix)):
                if vertex == w:
                    continue
                if self.matrix[vertex][w] > 0 or (not directed and self.matrix[w][vertex] > 0):
                    edgeState = self.BI if self.matrix[vertex][w] > 0 and self.matrix[w][vertex] > 0 else \
                        self.FWD if self.matrix[vertex][w] > 0 else self.REV
                    edgeSet.add((vertex, w, edgeState))
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
        for edge in self.edgeSet(v, directed=directed):
            if w in edge[:2]:
                return edge
        return None

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

    def setEdgeStates(self, config: list[int]):
        """Set the direction of each edge of the graph using the configuration"""
        edgeSet = list(self.edgeSet(directed=False))
        for idx, state in enumerate(config):
            self.setEdgeState(edgeSet[idx][0], edgeSet[idx][1], state, refreshState=False)
        self._refreshState()

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

    def addEdge(self, v: int, w: int, weight: int = 1, state: int = 0, refreshState=True):
        """Adds an edge between two vertices"""
        if v >= len(self.matrix) and w >= len(self.matrix):
            raise AttributeError("Both vertices cannot be new!")
        elif v >= len(self.matrix) or w >= len(self.matrix):
            self.matrix = np.append(self.matrix, [[0]*len(self.matrix)], 0)
            self.matrix = np.append(self.matrix, [[0]] * (len(self.matrix)), 1)
        self.setEdgeState(v, w, weight=weight, state=state, refreshState=refreshState)

    def setEdgeState(self, v: int, w: int, state: int = 0, weight: int = None, refreshState=True):
        """Sets as an edge as directed or undirected or change the direction of the edge"""
        weight = max(self.matrix[v][w], self.matrix[w][v]) if weight is None else weight

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
        size = len(self.matrix)
        selected_node = [0]*size
        no_edge = 0
        selected_node[0] = True
        tree = self.empty(size)

        while no_edge < size - 1:
            minimum = float('inf')
            a = 0
            b = 0
            for m in range(size):
                if selected_node[m]:
                    for n in range(size):
                        if not selected_node[n] and self.matrix[m][n]:
                            # not in selected and there is an edge
                            if minimum > self.matrix[m][n]:
                                minimum = self.matrix[m][n]
                                a = m
                                b = n
            tree.addEdge(a, b, self.matrix[a][b], refreshState=False)
            selected_node[b] = True
            no_edge += 1
        tree._refreshState()
        return tree

    def lend(self, divisor: Divisor, vertexes: list | np.ndarray | int, amount, forceLegal=False):
        """Lend from a vertex amount number of times.
        Borrows if amount is negative"""
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

    def visualize(self, divisor: Divisor = None, withWeights=False):
        """Creates a plot representation of the graph"""
        labels = {}
        for i in range(0, len(self.matrix)):
            labels[i] = f"{i}"+(f": {divisor[i]}" if divisor else "")
        G = nx.DiGraph()
        G.add_weighted_edges_from(self.edgeSet())
        # pos = nx.circular_layout(G)
        pos = nx.planar_layout(G)
        # pos = nx.kamada_kawai_layout(G)
        if withWeights:
            edgeLabels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edgeLabels)
        nx.draw_networkx(G, pos, labels=labels, node_size=700)
        plt.show()

    def copy(self):
        """Returns a copy of this graph"""
        return Graph(self.matrix)

    def __len__(self):
        return len(self.matrix)

    @classmethod
    def cycle(cls, size):
        """Returns a cycle graph of the given size"""
        graph = cls.empty(size)
        for i in range(0, size-1):
            graph.addEdge(i, i + 1, refreshState=False)
        graph.addEdge(0, size-1)
        return graph

    @classmethod
    def random(cls, size):
        """Returns a random graph of the given size"""
        graph = cls.empty(size)
        for i in range(0, size):
            for j in range(0, size):
                if i != j and not graph.getEdge(i, j, directed=False) \
                        and random.randint(0, 1) == 0:
                    graph.addEdge(i, j, refreshState=False)
                    graph.setEdgeState(i, j, random.randint(0, 2), refreshState=False)
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
    def _glue(cls, graph1: Graph, graph2: Graph, vertex1: int, vertex2: int, byVertex: bool, edgeState: int = BI):
        matrix1 = np.copy(graph1.matrix)
        matrix2 = np.copy(graph2.matrix)
        # Make desired rows and columns last and first for easy connection
        if vertex1 != len(matrix1)-1:
            matrix1[:, [len(matrix1)-1, vertex1]] = matrix1[:, [vertex1, len(matrix1)-1]]
            matrix1[[len(matrix1)-1, vertex1]] = matrix1[[vertex1, len(matrix1)-1]]
        if vertex2 != 0:
            matrix2[:, [0, vertex2]] = matrix2[:, [vertex2, 0]]
            matrix2[[0, vertex2]] = matrix2[[vertex2, 0]]

        m1Len = len(matrix1)
        offset = 1 if byVertex else 0
        glued = np.zeros((m1Len+len(matrix2)-offset, m1Len+len(matrix2)-offset))
        glued[0:0 + matrix1.shape[0], 0:0 + matrix1.shape[1]] = matrix1
        glued[m1Len-offset:m1Len-offset + matrix2.shape[0], m1Len-offset:m1Len-offset + matrix2.shape[1]] += matrix2

        if not byVertex:
            if edgeState == cls.BI or edgeState == cls.FWD:
                glued[m1Len-1][m1Len] = 1
            if edgeState == cls.BI or edgeState == cls.REV:
                glued[m1Len][m1Len-1] = 1

        return Graph(glued)

    @classmethod
    def glueByVertex(cls, graph1: Graph, graph2: Graph, vertex1: int, vertex2: int):
        return cls._glue(graph1, graph2, vertex1, vertex2, byVertex=True)

    @classmethod
    def glueByEdge(cls, graph1: Graph, graph2: Graph, vertex1: int, vertex2: int, state=0):
        return cls._glue(graph1, graph2, vertex1, vertex2, byVertex=False, edgeState=state)


def prettyCok(coKernel: tuple):
    cokStr = ""
    for factor in coKernel[1]:
        cokStr += f"\u2124_{factor} x "
    if len(coKernel[1]) == 0:
        cokStr += f"\u2124 x "
    if coKernel[2] > 0:
        cokStr += "\u2124"+(f"^{coKernel[2]}" if coKernel[2] > 1 else "")
    else:
        cokStr = cokStr[:-2]
    return cokStr


def allCyclics(graph: Graph, skipRotations=True, includeBi=True):
    """Generates all oriented permutations of a cyclic graph"""
    permutations = []
    config = [0 if includeBi else 1]*len(graph)
    idx = 0
    lastYield = time.time()

    def checkRotations():
        """Checks if the current permutation has been logged or if any of its rotations have"""
        if config in permutations:
            return False
        currCopy = config.copy()
        for _ in range(0, len(graph)-1):
            # Rotate by one vertex increment
            first = currCopy.pop(0)
            currCopy.append(first)
            if currCopy in permutations:
                return False
        return True

    def increment():
        """Increments the trinary number representing the configuration"""
        config[-1] += 1
        pointer = len(config)-1
        while config[pointer] == 3 and pointer > 0:
            config[pointer] = 0 if includeBi else 1
            pointer -= 1
            config[pointer] += 1

    for i in range(0, (3 if includeBi else 2)**len(graph)):
        if skipRotations and not checkRotations():
            increment()
            idx += 1
            continue
        permutations.append(config.copy())
        copy = graph.copy()
        copy.setEdgeStates(config)
        yield idx, config, copy, time.time()-lastYield
        lastYield = time.time()
        increment()
        idx += 1


def bruteCheckGraphs(graph: Graph, includeBi=True):
    """Checks manually to see if the cyclic graph of the given
    size has the appropriate invariant factors"""
    timer = time.time()

    sets = [None]*len(graph)
    globalIdx = 0
    found = []
    picAvg = 0
    yieldAvg = 0
    for idx, config, current, lastYield in allCyclics(graph, skipRotations=False, includeBi=includeBi):
        globalIdx += 1
        if globalIdx % 5000 == 0:
            logger.info(f"Checked up to {globalIdx} items...")
        chk = time.time()
        cok = current.pic()
        picAvg = (picAvg*(globalIdx-1)+time.time()-chk)/globalIdx
        yieldAvg = (yieldAvg*(globalIdx-1)+lastYield)/globalIdx
        # Account for trivial set if invariant factors are empty
        if len(cok[1]) == 0:
            cok[1].append(1)
        if sets[cok[1][0]-1] is None:
            if len(cok[1]) != 1:
                logger.warning("===> Large!")
            sets[cok[1][0]-1] = config
            found.append(globalIdx)
            logger.info(f"Found \u2124_{cok[1][0]} at: {globalIdx} {len(found)}/{len(graph)} "
                        f"({round(len(found)/len(graph)*100)}%), {config}")
            if len([elem for elem in sets if elem is not None]) == len(graph):
                # logger.info(f"Matches Guess: {set(guesses)==set(found)}")
                logger.info(f"Found sets from trivial through Z_{len(graph)}")
                logger.info(f"Finished after checking {globalIdx} permutations of {(3 if includeBi else 2) ** len(graph)} permutations"
                      f", ({round(globalIdx/(3 ** len(graph))*100, 5)}%) "
                      f"after {round((time.time()-timer), 3)}s")
                logger.info("--------")
                return True

    logger.info(f"Failed to find sets from trivial through Z_{len(graph)}, only found "
        f"{[idx for idx, elem in enumerate(sets) if elem is not None]}")
    logger.info(f"Failed after checking all {(3 if includeBi else 2) ** len(graph)} permutations"
        f", ({round(globalIdx / (3 ** len(graph)) * 100, 5)}%) "
        f"after {round((time.time() - timer), 3)}s")
    logger.info("--------")
    return False


def fireCycle(graph: Graph, divisor: Divisor, mutate=False):
    """Fires all vertices until none can fire anymore or the original
    configuration has been reached"""
    original = divisor.copy()
    if not mutate:
        divisor = divisor.copy()
    moveSet = graph.lend(divisor, list(range(len(divisor))), 1, forceLegal=True)
    while divisor != original and moveSet is not None and moveSet != [0]*len(moveSet):
        moveSet = graph.lend(divisor, list(range(len(divisor))), 1, forceLegal=True)

    return divisor


def greedy(graph: Graph, divisor: Divisor, mutate=False):
    """Finds an equivalent, effective divisor to the one given using
    the greedy algorithm"""
    if divisor.deg() < 0:
        return False, None, None
    marked = []
    moves = []
    if not mutate:
        divisor = divisor.copy()

    while not divisor.isEffective():
        if len(marked) < len(divisor):
            for i in range(0, len(divisor)):
                if divisor[i] < 0:
                    move = graph.lend(divisor, i, -1)
                    if move is None:
                        return False, None, None
                    moves.append(move)
                    if i not in marked:
                        marked.append(i)
                    break
        else:
            return False, None, None
    return True, moves, divisor


def dhar(graph: Graph, divisor: Divisor, source: int, burned: list[int] = None):
    """Determines if a divisor has a super-stable configuration about a source vertex"""
    if burned is None:
        burned = [source]

    for v in graph.adjacencySet(source):
        if v in burned:
            continue
        threats = 0
        for b in burned:
            if graph.getEdge(v, b):
                threats += 1
        if divisor[v] < threats:
            burned.append(v)
            dhar(graph, divisor, v, burned)

    return len(burned) == len(divisor), [i for i in range(0, len(divisor)) if i not in burned], burned


def qReduced(graph: Graph, divisor: Divisor, q=0, mutate=False):
    """Finds an equivalent, effective divisor to the one given using
    the q-reduced divisors algorithm"""
    if divisor.deg() < 0:
        return False, None, None

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
    isSuperstable = True
    while divisor[q] < 0 and isSuperstable:
        isSuperstable, legals, nonLegal = dhar(graph, divisor, q)
        graph.lend(divisor, legals, 1)

    if divisor[q] < 0:
        return False, None, None
    return True, moves, divisor


def debtReduction(graph: Graph, divisor: Divisor, q=0, mutate=False):
    """Employs the debt reduction trick to reduce the overall debt of a graph"""
    if not mutate:
        divisor = divisor.copy()
    firingScript = np.floor(np.matmul(np.linalg.inv(graph.reducedLaplacian(q)), divisor.config(q)))
    firedCfg = divisor.config(q) - np.matmul(graph.reducedLaplacian(q), firingScript)
    oldDeg = divisor.deg()
    for i in range(0, len(divisor)):
        if i != q:
            divisor[i] = firedCfg[i if i < q else i - 1]
    divisor[q] = oldDeg - np.sum(firedCfg)

    return divisor
