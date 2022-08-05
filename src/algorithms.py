import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Divisor:
    def __init__(self, chips: list[int]):
        self.rawChips = chips
        # vertical vector for multiplication
        self.divisor = np.array([chips]).transpose()
        self.iterPlace = 0

    def deg(self):
        """Return the degree of the divisor"""
        vsum = 0
        for v in self.divisor:
            vsum += v[0]
        return vsum

    def isEffective(self):
        """If the divisor is winning"""
        for v in self.divisor:
            if v[0] < 0:
                return False
        return True

    def config(self, vertex):
        return np.delete(self.divisor, vertex, axis=0)

    def copy(self):
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


class Utils:
    @classmethod
    def smithNormalForm(cls, matrix: np.ndarray):
        matrixClone = matrix.copy()

        def exchangeRows(other: np.ndarray, i: int, j: int):
            matrixClone[[i, j]] = matrixClone[[j, i]]
            other[[i, j]] = other[[j, i]]

        def exchangeCols(other: np.ndarray, i: int, j: int):
            matrixClone[:, [i, j]] = matrixClone[:, [j, i]]
            other[:, [i, j]] = other[:, [j, i]]

        def addRows(other: np.ndarray, i: int, j: int, scale=1):
            matrixClone[i, :] = (matrixClone[i, :] + scale * matrixClone[j, :])
            other[i, :] = (other[i, :] + scale * other[j, :])

        def addCols(other: np.ndarray, i: int, j: int, scale=1):
            matrixClone[:, i] = (matrixClone[:, i] + scale * matrixClone[:, j])
            other[:, i] = (other[:, i] + scale * other[:, j])

        def scaleRow(other: np.ndarray, i: int, scale):
            matrixClone[i, :] = scale * matrixClone[i, :]
            other[i, :] = scale * other[i, :]

        def scaleCol(other: np.ndarray, i: int, scale=1):
            matrixClone[:, i] = scale * matrixClone[:, i]
            other[:, i] = scale * other[:, i]

        m, n = matrixClone.shape

        def minAij(s):
            """Find the minimum non-zero element below and to the right of matrix[s][s]"""
            element = [s, s]
            globalMin = max(max([abs(x) for x in matrixClone[j][s:]]) for j in range(s, m))
            for i in (range(s, m)):
                for j in (range(s, n)):
                    if matrixClone[i][j] != 0 and abs(matrixClone[i][j]) <= globalMin:
                        element = [i, j]
                        globalMin = abs(matrixClone[i][j])
            return element

        def isLone(s):
            """Checks if matrix[s][s] is the only non-zero in col s below matrix[s][s] and the only
            non-zero in row s to the right of matrix[s][s]"""
            if [matrixClone[s][x] for x in range(s + 1, n) if matrixClone[s][x] != 0] + [matrixClone[y][s]
                    for y in range(s + 1, m) if matrixClone[y][s] != 0] == []:
                return True
            else:
                return False

        def findNonDivisible(s):
            """Finds the first element which is not divisible by matrix[s][s]"""
            for x in range(s + 1, m):
                for y in range(s + 1, n):
                    if matrixClone[x][y] % matrixClone[s][s] != 0:
                        return x, y
            return None

        p = np.identity(m)
        q = np.identity(n)
        for s in range(min(m, n)):
            while not isLone(s):
                # Get min location
                i, j = minAij(s)
                exchangeRows(p, s, i)
                exchangeCols(q, s, j)
                for x in range(s + 1, m):
                    if matrixClone[x][s] != 0:
                        k = matrixClone[x][s] // matrixClone[s][s]
                        addRows(p, x, s, -k)
                for x in range(s + 1, n):
                    if matrixClone[s][x] != 0:
                        k = matrixClone[s][x] // matrixClone[s][s]
                        addCols(q, x, s, -k)
                if isLone(s):
                    res = findNonDivisible(s)
                    if res:
                        x, y = res
                        addRows(p, s, x, 1)
                    else:
                        if matrixClone[s][s] < 0:
                            scaleRow(p, s, -1)

        return np.matmul(p, np.matmul(matrix, q)), p, q

    @classmethod
    def coKernel(cls, matrix: np.ndarray, divisor: Divisor | np.ndarray = None):
        """Returns the polynomial, invariant factors, and rank of the coKernel of the given matrix"""
        smith, p, q = cls.smithNormalForm(matrix)
        print(smith)
        if not divisor:
            product = p
        else:
            product = np.matmul(p, divisor)
        infs = []
        invFactors = []
        delModifier = 0
        for i in range(len(smith)):
            if smith[i][i] == 1 or np.all((smith[i] == 0)):
                product = np.delete(product, i + delModifier, axis=0)
                delModifier -= 1
                if np.all((smith[i] == 0)):
                    infs.append(float("inf"))
            else:
                invFactors.append(smith[i][i])
        product = [np.atleast_1d(layer).tolist() for layer in product]
        if len(infs) > 0:
            product.append(infs)

        # polynomial, invariant factors, rank
        return product, invFactors, len(infs)


class Graph:
    BIDIRECTIONAL = 0
    FORWARD = 1
    BACKWARDS = 2

    def __init__(self, adjacency: list[list[int]]):
        self.matrix = adjacency.copy()
        self.laplacian = self._makeLaplacian()

    def _makeLaplacian(self):
        """Create the laplacian matrix for the graph"""
        laplacian = np.zeros((len(self.matrix),)*2)
        for i in range(0, len(laplacian)):
            for j in range(0, len(laplacian)):
                laplacian[i][j] = self.degree(i) if i == j else -self.commonEdges(i, j)
        return laplacian

    def reducedLaplacian(self, vertex):
        return np.delete(np.delete(self.laplacian, vertex, axis=0), vertex, axis=1)

    def edgeSet(self, vertex=-1, directed=True):
        """Returns either the edge set for a vertex or the edge set for the whole graph"""
        edgeSet = set()
        if vertex < 0:
            """return [(v, w, self.matrix[v][w]) for w in range(0, len(self.matrix))
                        for v in range(0, len(self.matrix)) if v != w and self.matrix[v][w] > 0]"""
            for v in range(0, len(self.matrix)):
                for w in range(0, len(self.matrix)):
                    if not directed and v > w:
                        continue
                    if self.matrix[v][w] > 0 or (not directed and self.matrix[w][v] > 0):
                        edgeSet.add((v, w, self.matrix[v][w]))
        else:
            for w in range(0, len(self.matrix)):
                if not directed and vertex > w:
                    continue
                if self.matrix[vertex][w] > 0 or (not directed and self.matrix[w][vertex] > 0):
                    edgeSet.add((vertex, w, self.matrix[vertex][w]))
        return edgeSet
        """return [(vertex, w, self.matrix[vertex][w])
                for w in range(0, len(self.matrix)) if vertex != w and self.matrix[vertex][w] > 0]"""

    def commonEdges(self, v, w):
        return self.matrix[v][w]

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

    def hasEdge(self, v, w, directed=True):
        """Return whether two vertices share an edge"""
        edges = self.edgeSet(v, directed)
        return (v, w) in edges

    def setEdgeStates(self, config: list[int]):
        edgeDict = {i: edge for i, edge in enumerate(self.edgeSet(directed=False))}
        for idx, state in enumerate(config):
            self.setEdgeState(edgeDict[idx][0], edgeDict[idx][1], state, refreshLaplacian=False)
            self.laplacian = self._makeLaplacian()

    def setEdgeState(self, v: int, w: int, state: int, refreshLaplacian=True):
        """Sets as an edge as directed or undirected or change the direction of the edge"""
        weight = max(self.matrix[v][w], self.matrix[w][v])
        if state == 0:
            self.matrix[v][w] = weight
            self.matrix[w][v] = weight
        elif state == 1:
            self.matrix[v][w] = weight
            self.matrix[w][v] = 0
        elif state == 2:
            self.matrix[v][w] = 0
            self.matrix[w][v] = weight
        if refreshLaplacian:
            self.laplacian = self._makeLaplacian()

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

    def jac(self, divisor: Divisor = None, vertex=0):
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
        pos = nx.circular_layout(G)
        if withWeights:
            edgeLabels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edgeLabels)
        nx.draw_networkx(G, pos, labels=labels, node_size=700)
        plt.show()

    def copy(self):
        return Graph(self.matrix)

    def __len__(self):
        return len(self.matrix)


def allGraphs(graph: Graph):
    permutations = []
    current = [0]*len(graph)

    def checkRotations():
        """Checks if the current permutation has been logged or if any of its rotations have"""
        if current in permutations:
            return False
        currCopy = current.copy()
        for _ in range(0, len(graph)-1):
            # Rotate by one vertex increment
            first = currCopy.pop(0)
            currCopy.append(first)
            if currCopy in permutations:
                return False
        return True

    def increment():
        current[-1] += 1
        pointer = len(current)-1
        while current[pointer] == 3 and pointer > 0:
            current[pointer] = 0
            pointer -= 1
            current[pointer] += 1

    for i in range(0, 3**len(graph)):
        if not checkRotations():
            increment()
            continue
        permutations.append(current.copy())
        copy = graph.copy()
        copy.setEdgeStates(current)
        yield copy
        increment()

    print("Finished")


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
            if graph.hasEdge(v, b):
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

"""
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
                lIdx += 1"""


"""

        def choosePivot(lastJ):
            for j in range(0, len(matrix[0])):
                for k in range(0 if lastJ == -1 else lastJ + 1, len(matrix)):
                    if k != 0:
                        return k, j
        # Step One
        lastJ = -1
        for t in range(0, len(matrix)):
            j, k = choosePivot(lastJ)
            if matrix[t][lastJ] == 0:
                # Exchange rows
                exchangeRows(t, k)
            lastJ = j
            for k in range(0, len(matrix)):
                if matrix[t][j] % matrix[k][j] != 0:
                    gcd = math.gcd(matrix[t][j], matrix[k][j])"""
