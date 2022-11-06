from __future__ import annotations
from dataclasses import dataclass, asdict
import multiprocessing
import time
import queue as QueueClass
import traceback
from game import *
import matplotlib.pyplot as plt
from utils import *


class Orientor:
    paramDict = {}
    processing = False

    @classmethod
    def init(cls, graph: Graph, includeBi=True, skipRotations=False, includePaths: tuple = None):
        edges = len(graph.edgeSet(directed=False))
        cls.paramDict = {"graph": graph, "edges": edges, "totalProcesses": multiprocessing.cpu_count(),
                         "config": [0 if includeBi else 1] * edges, "includeBi": includeBi,
                         "skipRotations": skipRotations, "includePaths": includePaths}

    @classmethod
    def allOrientations(cls):
        while True:
            if not cls.processing:
                cls.processing = True
                paramList = []
                for i in range(cls.paramDict["totalProcesses"]):
                    params = Params(i, **cls.paramDict)
                    paramList.append(params)
                pool.map_async(cls._allOrientations, paramList)
                time.sleep(5)

            try:
                res = queue.get(block=True, timeout=5)
                yield res
            except QueueClass.Empty:
                cls.processing = False
                logger.info("Processing ended successfully")
                break
            except Exception as e:
                logger.error(f"Error {e} {traceback.format_exc()}")
                break

    @classmethod
    def _allOrientations(cls, params):
        """Generates all oriented permutations of a cyclic graph"""
        graphIdx = [0]
        graphLimit = (3 if params.includeBi else 2) ** params.edges
        lastYield = time.time()
        permutations = []
        # print(f"Starting {procId}")

        def increment(value: int = 1):
            """Increments the trinary number representing the configuration"""
            for _ in range(value):
                params.config[-1] += 1
                graphIdx[0] += 1
                pointer = len(params.config) - 1
                while params.config[pointer] == 3 and pointer > 0:
                    params.config[pointer] = 0 if params.includeBi else 1
                    pointer -= 1
                    params.config[pointer] += 1

        def checkRotations():
            """Checks if the current permutation has been logged or if any of its rotations have"""
            if params.config in permutations:
                return False
            currCopy = params.config.copy()
            for _ in range(0, len(params.config) - 1):
                # Rotate by one vertex increment
                first = currCopy.pop(0)
                currCopy.append(first)
                if currCopy in permutations:
                    return False
            return True

        # Offset to desired index
        increment(params.idx)
        while True:
            if params.skipRotations and not checkRotations():
                increment(params.totalProcesses)
                continue
            if params.skipRotations:
                permutations.append(params.config.copy())
            copy = params.graph.copy()
            copy.setEdgeStates(params.config)
            paths = copy.countPaths() if params.includePaths else 0
            if not params.includePaths or (params.includePaths and paths in params.includePaths):
                queue.put((graphIdx[0], params.config, copy, time.time() - lastYield, paths))
            lastYield = time.time()
            increment(params.totalProcesses)
            if graphIdx[0] > graphLimit:
                break


@dataclass
class Params:

    def __init__(self, idx, **kwargs):
        self.idx = idx
        self.graph = kwargs["graph"]
        self.edges = kwargs["edges"]
        self.config = kwargs["config"]
        self.skipRotations = kwargs["skipRotations"]
        self.includePaths = kwargs["includePaths"]
        self.includeBi = kwargs["includeBi"]
        self.totalProcesses = kwargs["totalProcesses"]

    def __repr__(self):
        return f"Params({self.idx})"


lock = multiprocessing.Lock()
queue = multiprocessing.Queue()
pool = multiprocessing.Pool()


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
    Orientor.init(graph, skipRotations=skipRotations,
            includeBi=includeBi, includePaths=includePaths)
    for idx, config, current, lastYield, paths, in Orientor.allOrientations():
        globalIdx += 1
        if globalIdx % 500 == 0:
            logger.info(f"({round(globalIdx/((3 if includeBi else 2) ** Orientor.paramDict['edges'])*100, 2)}%) "
                f"Checked up to {globalIdx} items...")
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
        # logger.info(f"Path factors: {pathInvFactors}")

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
    total = 0
    for factor, occurrence in zip(factors, occurrences):
        factorStr += f"\u2124_{factor}: {occurrence}, "
        total += occurrence
    logger.info(factorStr[:-1])
    logger.info(f"Total graphs processed: {total}")
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
    orientor = Orientor(graph, skipRotations=skipRotations, includeBi=includeBi)
    for idx, config, current, lastYield, _ in Orientor.allOrientations(orientor):
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
