import datetime
import logging

from algorithms import *
import time


# Do komplete 4 and 5

def testSrcSnk():
    graph = Graph.cycle(4)
    graph.forceFlow(0)
    graph.visualize()
    print(graph.jac(2))
    graph.forceFlow(0, makeSink=False)
    graph.visualize()
    print(graph.jac(2))


def testBruteForce(rng=(3, 10)):
    cokDict = {i: bruteCheckGraphs(Graph.cycle(i), includeBi=True) for i in range(*rng)}
    logger.info(f"CoKernels {cokDict}")


def testSinkSource(graph: Graph):
    print(Utils.prettyCok(graph.pic()))
    graph.forceFlow(0, makeSink=True)
    print(Utils.prettyCok(graph.pic()))
    graph.forceFlow(0, makeSink=False)
    print(Utils.prettyCok(graph.pic()))


def testAllJacs():
    jacs = {}
    times = []
    for i in range(5, 6):
        timeCheck = time.time()
        jacs[i] = []
        graph = Graph.cycle(i)
        graph.visualize(title=f"Pic(G)={Utils.prettyCok(graph.pic())}")
        jacs[i].append(graph.pic()[1][0])
        graph.setEdgeState(1, 2, state=1)
        graph.setEdgeState(2, 3, state=2)
        if i > 4:
            for j in range(3, i - 1):
                graph.setEdgeState(j, j + 1, state=2)
                jacs[i].append(graph.pic()[1][0])
                graph.visualize(title=f"Pic(G)={Utils.prettyCok(graph.pic())}")
        graph.setEdgeState(i - 1, 0, state=2)
        jacs[i].append(graph.pic()[1][0])
        graph.visualize(title=f"Pic(G)={Utils.prettyCok(graph.pic())}")

        graph.setEdgeState(0, 1, state=2)
        jacs[i].append(graph.pic()[1][0])
        graph.visualize(title=f"Pic(G)={Utils.prettyCok(graph.pic())}")

        graph.setEdgeState(1, 2, state=2)
        jacs[i].append(1 if len(graph.pic()[1]) == 0 else graph.pic()[1][0])
        graph.visualize(title=f"Pic(G)={Utils.prettyCok(graph.pic())}")
        check = jacs[i] == [i for i in range(i, 0, -1)]
        print(f"Check for {i}: {check} after {time.time() - timeCheck}")
        times.append((i, round(time.time() - timeCheck, 3)))
        if not check:
            print("/\\ Error, not checked!")
    print(jacs)
    print(times)


def wheelJacs(n):
    if n == 0:
        return 4, 4
    elif n == 1:
        return 3, 15

    if n % 2 == 0:
        num = 5 * wheelJacs(n - 1)[0] - wheelJacs(n - 2)[0]
        return num, num
    num = wheelJacs(n - 1)[0] - wheelJacs(n - 2)[0]
    return num, 5 * num


def testPseudoTree(glueByVertex=True):
    Utils.setLogged(True)
    LoggedMatrixOps.resetLogging()
    cycle = Graph.cycle(5)
    cycle.setEdgeState(0, 1, 1)
    cycle.setEdgeState(1, 2, 1)
    cycle.setEdgeState(2, 3, 1)
    cycle.setEdgeState(3, 4, 2)
    # cycle.visualize()
    audit = cycle.auditEdges()
    print(f"Cycle sources: {audit[0]}, Cycle sinks: {audit[1]}")
    print(f"Cycle Picard: {Utils.prettyCok(cycle.pic())}")
    adjacency = [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ]
    stick = Graph(adjacency)
    audit = stick.auditEdges()
    print(f"Tree sources: {audit[0]}, Tree sinks: {audit[1]}")
    print(f"Tree picard: {Utils.prettyCok(stick.pic())}")
    gluePoint = 4
    if glueByVertex:
        glued = Graph.glueByVertex((cycle, gluePoint), (stick, 0))
    else:
        glued = Graph.glueByEdge((cycle, gluePoint), (stick, 0), state=Graph.FWD)
    # glued.visualize()
    print(f"Pseudo-tree picard: {Utils.prettyCok(glued.pic())}")
    check = time.time()
    LoggedMatrixOps.resetLogging()
    finished = Utils.smithNormalForm(glued.laplacian)
    logger.info(f"Operations: {LoggedMatrixOps.opNumber}")
    logger.info(f"Pseudo-tree picard: {finished[0]}")
    logger.info(f"Monolithic Time diff: {time.time()-check}")
    logging.info("======================================================")
    check = time.time()
    LoggedMatrixOps.resetLogging()
    DEBUG["snfRange"] = (0, 5)

    def prepare(matrix):
        mimic = np.identity(len(matrix))
        MatrixOps.addRows(matrix, mimic, 5, 4)
        MatrixOps.addCols(matrix, mimic, 4, 5)

    DEBUG["snfPrepare"] = prepare
    partial = Utils.smithNormalForm(glued.laplacian)
    DEBUG["snfPrepare"] = None
    DEBUG["snfRange"] = (5, 9)
    final = Utils.smithNormalForm(partial[0])
    DEBUG["snfRange"] = None
    finishedd = Utils.smithNormalForm(final[0])
    logger.info(f"Operations: {LoggedMatrixOps.opNumber}")
    logger.info(f"Pseudo-tree picard: {finishedd[0]}")
    logger.info(f"Dispersed Time diff: {time.time() - check}")
    logger.info(f"Match? {finishedd[0].all() == finished[0].all()}")
    audit = glued.auditEdges()
    print(f"Pseudo-tree sources: {audit[0]}, Pseudo-tree sinks: {audit[1]}")


def cycleOrientation(size: int):
    cycle = Graph.cycle(size)
    cycle.setEdgeState(0, 1, 0)
    cycle.setEdgeState(1, 2, 0)
    cycle.setEdgeState(2, 3, 0)
    cycle.setEdgeState(3, 4, 1)
    cycle.setEdgeState(4, 5, 2)
    """cycle.setEdgeState(5, 6, 0)
    cycle.setEdgeState(6, 7, 0)
    cycle.setEdgeState(7, 8, 0)
    cycle.setEdgeState(8, 9, 0)
    cycle.setEdgeState(9, 10, 0)"""
    cycle.setEdgeState(size - 1, 0, 0)
    cycle.visualize()
    print(f"Cycle Picard of size {size}: {Utils.prettyCok(cycle.pic())}")
    print(f"Cycle paths: {cycle.countPaths()}")


def wheelOrientation(size: int):
    wheel = Graph.wheel(size, direction=0, spokeDirection=0)
    # wheel.setEdgeState(1, 2, 1)
    # wheel.setEdgeState(2, 3, 2)
    # wheel.setEdgeState(3, 4, 1)
    # wheel.setEdgeState(4, 5, 1)
    """wheel.setEdgeState(5, 6, 0)
    wheel.setEdgeState(6, 7, 0)
    wheel.setEdgeState(7, 8, 0)
    wheel.setEdgeState(8, 9, 0)
    wheel.setEdgeState(9, 10, 0)"""
    # wheel.setEdgeState(size - 1, 1, 1)
    # wheel.visualize(positions={0: (0, 0)})
    # wheel.visualize()
    print(f"Wheel Picard of size {size}: {Utils.prettyCok(wheel.pic())}")


def picTimes(size):
    Utils.setLogged(True)
    LoggedMatrixOps.resetLogging()
    wheel = Graph.wheel(size)
    cycle = Graph.cycle(size)
    complete = Graph.complete(size)
    cycle2 = Graph.cycle(size)
    # glued = Graph.glueByEdge((wheel, 0), (cycle, 0), (complete, 0), (cycle2, 0))
    glued = Graph.glueByEdge((wheel, 0), (cycle, 0))
    check = time.time()
    finished = Utils.subSmith(glued.laplacian, 0)
    logger.info(f"({2*size}) Whole Operations: {LoggedMatrixOps.opNumber}")
    wholeDiff = time.time() - check
    logger.info(f"({2*size}) Whole Time diff: {wholeDiff}")

    check = time.time()
    LoggedMatrixOps.resetLogging()
    finished = Utils.subSmith(glued.laplacian, 1)
    logger.info(f"({2*size}) Split Operations: {LoggedMatrixOps.opNumber}")
    splitDiff = time.time() - check
    logger.info(f"({2*size}) Split Time diff: {splitDiff}")

    return wholeDiff, splitDiff


if __name__ == "__main__":
    start = time.time()
    logger.info(f"=====\nEntering program at {datetime.datetime.fromtimestamp(start).strftime('%H:%M:%S')}\n=====")
    """adjacency = [
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [1, 1, 0, 1],
        [0, 0, 0, 0],
    ]"""
    """adjacency = [
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    divisor = Divisor([16, -4, -5, 0])
    graph = Graph(adjacency)
    graph.visualize(divisor)"""
    Utils.setLogged(True)
    maxGraph = 6
    minGraph = 5
    # [plotFactors('cycle', (size, maxGraph), includePaths=(2,), bySize=False) for size in range(minGraph, maxGraph)]
    # logger.info(f"Total ops: {LoggedMatrixOps.opNumber}, weighted ops: {LoggedMatrixOps.weightedOpNumber}")

    def cycleWheel(cycleSize: int):
        LoggedMatrixOps.resetLogging()
        cycle = Graph.cycle(cycleSize)
        logger.info(Utils.prettyCok(cycle.pic()))
        opsList = LoggedMatrixOps.ops
        LoggedMatrixOps.resetLogging()
        wheel = Graph.wheel(cycleSize+1, direction=0, spokeDirection=1)
        logger.info(Utils.prettyCok(wheel.pic()))
        return LoggedMatrixOps.operate(wheel.laplacian, opsList, lambda x: x + 1)


    cycleSize = 9
    cycle = Graph.cycle(cycleSize)
    cycle.setEdgeState(1, 2, Graph.FWD)
    cycle.setEdgeState(2, 3, Graph.REV)
    cycle.setEdgeState(3, 4, Graph.REV)
    logger.info(Utils.prettyCok(cycle.pic()))
    wheel = Graph.wheel(cycleSize + 1, direction=0, spokeDirection=1)
    wheel.setEdgeState(0, 3, Graph.REV)
    wheel.setEdgeState(0, 5, Graph.REV)
    # wheel.setEdgeState(1, 2, Graph.FWD)
    # wheel.setEdgeState(2, 3, Graph.REV)
    # wheel.setEdgeState(3, 4, Graph.REV)

    logger.info(Utils.prettyCok(wheel.pic()))
    # cokDict = {i: allStats(Graph.cycle(i), skipRotations=False) for i in range(8, 9)}
    # logger.info(cokDict)
    # testBruteForce((3, 40))
    # graph = Graph(adjacency)
    # graph.visualize()
    # print(graph.auditEdges())
    # testPseudoTree(glueByVertex=False)
    # times = [picTimes(size) for size in range(3, 15)]
    # testAllJacs()
    # Utils.setVerbose(True)
    """for i in range(10, 11):
        logger.info(f"Cycle graph of size {i}")
        cycle = Graph.cycle(i, direction=Graph.REV)
        cycle.setEdgeState(i - 9, i - 10, Graph.FWD)
        cycle.setEdgeState(i - 8, i - 9, Graph.REV)
        cycle.setEdgeState(i - 8, i - 7, Graph.FWD)
        cycle.setEdgeState(i - 7, i - 6, Graph.BI)
        cycle.setEdgeState(i - 6, i - 5, Graph.FWD)
        cycle.setEdgeState(i - 5, i - 4, Graph.REV)
        cycle.setEdgeState(i - 4, i - 3, Graph.REV)
        cycle.setEdgeState(i - 3, i - 2, Graph.BI)
        cycle.setEdgeState(i - 2, i - 1, Graph.REV)
        cycle.setEdgeState(i - 1, 0, Graph.FWD)
        # cycle.visualize()
        Utils.setLogged(True)
        logger.info(f"Cycle graph of size {i} Pic: {Utils.prettyCok(cycle.pic())}")"""

    """Utils.setLogged(True)
    cycle = Graph.cycle(10)
    LoggedMatrixOps.startOps()
    Utils.smithNormalForm(cycle.laplacian)
    ops = LoggedMatrixOps.opNumber

    mimic = np.identity(len(cycle.laplacian))
    LoggedMatrixOps.startOps()
    prepped = Utils.prepSmith(cycle.laplacian, mimic, mimic, ops=LoggedMatrixOps)
    Utils.smithNormalForm(prepped, resetLogCount=False)
    preppedOps = LoggedMatrixOps.opNumber
    logger.info(f"Unprepared: {ops}, prepared: {preppedOps}")"""
    # cycleOrientation(6)
    # [wheelOrientation(i) for i in range(6, 16)]
    """[wheelOrientation(i) for i in range(3, 21)]
    for i in range(0, 21):
        logger.info(f"{i}: {wheelJacs(i)}")"""

    """n = 800
    cycle = Graph.cycle(n)
    print(f"Cycle Picard of size {n}: {Utils.prettyCok(cycle.pic())}")"""
    """for i in range(1, 16):
        network = Graph.network([i, 100])
        # print(f"{i} Picard: {Utils.prettyCok(network.pic())}")
        print(f"{i} Picard: {Utils.prettyCok(network.pic(), compact=True)}")"""
    """Graph.glueByEdge(Graph.cycle(5), Graph([[0, 1, 1], [1, 0, 0], [1, 0, 0]]), vertex1=4, vertex2=0).visualize(
        title="A Pseudo-Tree Graph", positions={0: [-.5, 0], 4: [.5, 0], 1: [-.9, .5], 3: [.9, .5], 5: [0, -.5]}
    )"""
    """cycle = Graph.cycle(7)
    cycle.setEdgeState(0, 1, 0)
    cycle.setEdgeState(1, 2, 1)
    cycle.setEdgeState(2, 3, 2)
    cycle.setEdgeState(3, 4, 2)
    cycle.setEdgeState(4, 5, 2)
    cycle.setEdgeState(5, 6, 0)
    cycle.setEdgeState(6, 0, 0)
    cycle.visualize()
    # lap = cycle.laplacian
    # snf = Utils.smithNormalForm(cycle.laplacian)
    print(f"Cycle Picard of size {4}: {Utils.prettyCok(cycle.pic())}")"""
    """Utils.setVerbose(True)
    cycle = Graph.cycle(4)
    cycle.setEdgeState(0, 1, 1)
    cycle.setEdgeState(1, 2, 2)
    cycle.setEdgeState(2, 3, 0)
    cycle.setEdgeState(3, 0, 0)
    # lap = cycle.laplacian
    # snf = Utils.smithNormalForm(cycle.laplacian)
    print(f"Cycle Picard of size {4}: {Utils.prettyCok(cycle.pic())}")

    cycle.setEdgeState(3, 0, weight=0)
    cycle.setEdgeState(2, 3, 2)
    cycle.addEdge(3, 4, state=0)
    cycle.setEdgeState(4, 0, 0)
    # lap2 = cycle.laplacian
    # snf2 = Utils.smithNormalForm(cycle.laplacian)
    print(f"Cycle Picard of size {5}: {Utils.prettyCok(cycle.pic())}")

    cycle.setEdgeState(4, 0, weight=0)
    cycle.setEdgeState(3, 4, 2)
    cycle.addEdge(4, 5, state=0)
    cycle.setEdgeState(5, 0, 0)
    # lap3 = cycle.laplacian
    # snf3 = Utils.smithNormalForm(cycle.laplacian)
    print(f"Cycle Picard of size {6}: {Utils.prettyCok(cycle.pic())}")
    # 
    
    # cycleOrientation(4)"""
    """adjacency = [
        [0, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
    ]
    stick = Graph(adjacency)
    stick.visualize(title="A Tree Graph", positions={1: [.8, -.4], 2: [.9, .9], 0: [0, .5]})
    audit = stick.auditEdges()
    print(f"Tree sources: {audit[0]}, Tree sinks: {audit[1]}")
    print(f"Tree picard: {Utils.prettyCok(stick.pic())}")"""
    """graph = Graph([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
    ])
    #graph.addEdge(4, 5, state=1)
    #graph.addEdge(4, 6, state=1)
    #graph.addEdge(5, 7, state=2)
    graph.visualize()
    print(Utils.prettyCok(Utils.coKernel(graph.laplacian)))"""
    """graph = Graph.complete(3)
    testSinkSource(graph)"""
    # print(graph.jac(divisor))
    # print(graph.pic(divisor))
    """for v in range(len(graph)):
        print(f"Jacobian at vertex {v}:", graph.jac(vertex=v))
    print("Picard Group:", graph.pic())"""
    end = time.time()
    logger.info(f"=====\nExiting program at {datetime.datetime.fromtimestamp(end).strftime('%H:%M:%S')}"
          f"\nElapsed time: {round(end-start, 3)}s\n=====")

