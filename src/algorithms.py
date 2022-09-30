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

    def visualize(self, divisor: Divisor = None, withWeights=False, title=None, positions=None):
        """Creates a plot representation of the graph"""
        labels = {}
        for i in range(0, len(self.matrix)):
            labels[i] = f"{i}"+(f": {divisor[i]}" if divisor else "")
        G = nx.DiGraph()
        G.add_weighted_edges_from(self.edgeSet())
        pos = nx.circular_layout(G)
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


def allOrientations(graph: Graph, skipRotations=True, includeBi=True, includePaths: tuple = None):
    """Generates all oriented permutations of a cyclic graph"""
    permutations = []
    edges = len(graph.edgeSet(directed=False))
    config = [0 if includeBi else 1]*edges
    idx = 0
    lastYield = time.time()

    def checkRotations():
        """Checks if the current permutation has been logged or if any of its rotations have"""
        if config in permutations:
            return False
        currCopy = config.copy()
        for _ in range(0, len(config)-1):
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

    for i in range(0, (3 if includeBi else 2)**edges):
        if skipRotations and not checkRotations():
            increment()
            idx += 1
            continue
        permutations.append(config.copy())
        copy = graph.copy()
        copy.setEdgeStates(config)
        paths = copy.countPaths()
        if not includePaths or (includePaths and paths in includePaths):
            yield idx, config, copy, time.time()-lastYield, paths
        lastYield = time.time()
        increment()
        idx += 1


def plotFactors(graphForm: str, sizeRange: tuple, includeBi=True, skipRotations=False,
                includePaths: tuple = None, bySize=True):
    if bySize:
        for i in range(*sizeRange):
            factors, occurrences = allStats(Graph.build(graphForm, i), includeBi, skipRotations, includePaths)
            plt.plot(factors, occurrences, marker='o', label=f'Graph {i}')
            plt.title("Invariant Factors by Graph Size")
    else:
        factors, occurrences, pathFactors, pathOccurrences = allStats(Graph.build(graphForm, sizeRange[0]),
            includeBi, skipRotations, (0, 1, *range(2, sizeRange[0]+1, 2)))
        for paths in pathFactors:
            plt.plot(pathFactors[paths], pathOccurrences[paths], marker='o', label=f'{paths} Paths')
            plt.title(f"Invariant Factors by Path Number for a Size {sizeRange[0]} Graph")

    plt.xlabel('Inv Factors')
    plt.ylabel('Occurrences')
    plt.legend()
    plt.show()


def allStats(graph: Graph, includeBi=True, skipRotations=False, includePaths: tuple = None):
    timer = time.time()
    logger.info(f"===> Checking graph of size {len(graph)} in series with {includePaths} paths")
    invFactors = {}
    pathInvFactors = {}
    globalIdx = 0
    picAvg = 0
    yieldAvg = 0
    for idx, config, current, lastYield, paths, in allOrientations(graph, skipRotations=skipRotations,
            includeBi=includeBi, includePaths=includePaths):
        globalIdx += 1
        if globalIdx % 5000 == 0:
            logger.info(f"Checked up to {globalIdx} items...")
        chk = time.time()
        cok = current.pic()
        picAvg = (picAvg * (globalIdx - 1) + time.time() - chk) / globalIdx
        yieldAvg = (yieldAvg * (globalIdx - 1) + lastYield) / globalIdx
        # Account for trivial set if invariant factors are empty
        if len(cok[1]) == 0:
            cok[1].append(1)
        if paths not in pathInvFactors:
            pathInvFactors[paths] = {}
        if cok[1][0] not in invFactors:
            invFactors[cok[1][0]] = 0
        if cok[1][0] not in pathInvFactors[paths]:
            pathInvFactors[paths][cok[1][0]] = 0
        invFactors[cok[1][0]] += 1
        pathInvFactors[paths][cok[1][0]] += 1

    if len(invFactors) == 0:
        logger.warning("No graphs matched selection.  Exiting.")
        return [], []

    graphLen = max((*invFactors.keys(), len(graph)))
    factors = [i for i in range(1, graphLen+1)]
    pathFactors = {paths: [i for i in range(1, graphLen + 1)] for paths in pathInvFactors}
    occurrences = [0]*graphLen
    pathOccurrences = {paths: [0] * graphLen for paths in pathInvFactors}
    for factor in invFactors:
        occurrences[factor-1] = invFactors[factor]

    for path in pathInvFactors:
        for factor in pathInvFactors[path]:
            pathOccurrences[path][factor - 1] = pathInvFactors[path][factor]
    factorStr = "Frequencies: "
    for factor, occurrence in zip(factors, occurrences):
        factorStr += f"\u2124_{factor}: {occurrence}, "
    logger.info(factorStr[:-1])
    logger.info(f"Finished after {round((time.time()-timer), 3)}s")

    return factors, occurrences, pathFactors, pathOccurrences


def bruteCheckGraphs(graph: Graph, includeBi=True, skipRotations=False):
    """Checks manually to see if the cyclic graph of the given
    size has the appropriate invariant factors"""
    timer = time.time()
    logger.info(f"===> Checking graph of size {len(graph)} in series")
    edges = len(graph.edgeSet(directed=False))
    sets = [None]*edges
    globalIdx = 0
    found = []
    picAvg = 0
    yieldAvg = 0
    for idx, config, current, lastYield, _ in allOrientations(graph, skipRotations=skipRotations, includeBi=includeBi):
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
        if cok[1][0]-1 < len(sets) and sets[cok[1][0]-1] is None:
            sets[cok[1][0]-1] = config
            found.append(globalIdx)
            logger.info(f"Found \u2124_{cok[1][0]} at: {globalIdx} {len(found)}/{len(graph)} "
                        f"({round(len(found)/len(graph)*100)}%), {config}")
            if len([elem for elem in sets if elem is not None]) == len(graph):
                logger.info(f"Found sets from trivial through Z_{len(graph)}")
                logger.info(f"Finished after checking {globalIdx} ({idx} with skipped rotations) permutations "
                      f"of {(3 if includeBi else 2) ** edges} permutations"
                      f", ({round(globalIdx/(3 ** edges)*100, 5)}%) "
                      f"after {round((time.time()-timer), 3)}s")
                logger.info("--------")
                return True

    logger.info(f"Failed to find sets from trivial through Z_{len(graph)}, only found "
        f"{[idx for idx, elem in enumerate(sets) if elem is not None]}")
    logger.info(f"Failed after checking all {(3 if includeBi else 2) ** edges} permutations"
        f" after {round((time.time() - timer), 3)}s")
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
