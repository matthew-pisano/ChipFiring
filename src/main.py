import datetime

from algorithms import *
from src import results
from utils import logger


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
    logger.info(cokDict)


def testSinkSource(graph: Graph):
    print(prettyCok(graph.pic()))
    graph.forceFlow(0, makeSink=True)
    print(prettyCok(graph.pic()))
    graph.forceFlow(0, makeSink=False)
    print(prettyCok(graph.pic()))


def testAllJacs():
    jacs = {}
    times = []
    for i in range(5, 6):
        timeCheck = time.time()
        jacs[i] = []
        graph = Graph.cycle(i)
        graph.visualize(title=f"Pic(G)={prettyCok(graph.pic())}")
        jacs[i].append(graph.pic()[1][0])
        graph.setEdgeState(1, 2, state=1)
        graph.setEdgeState(2, 3, state=2)
        if i > 4:
            for j in range(3, i - 1):
                graph.setEdgeState(j, j + 1, state=2)
                jacs[i].append(graph.pic()[1][0])
                graph.visualize(title=f"Pic(G)={prettyCok(graph.pic())}")
        graph.setEdgeState(i - 1, 0, state=2)
        jacs[i].append(graph.pic()[1][0])
        graph.visualize(title=f"Pic(G)={prettyCok(graph.pic())}")

        graph.setEdgeState(0, 1, state=2)
        jacs[i].append(graph.pic()[1][0])
        graph.visualize(title=f"Pic(G)={prettyCok(graph.pic())}")

        graph.setEdgeState(1, 2, state=2)
        jacs[i].append(1 if len(graph.pic()[1]) == 0 else graph.pic()[1][0])
        graph.visualize(title=f"Pic(G)={prettyCok(graph.pic())}")
        check = jacs[i] == [i for i in range(i, 0, -1)]
        print(f"Check for {i}: {check} after {time.time() - timeCheck}")
        times.append((i, round(time.time() - timeCheck, 3)))
        if not check:
            print("/\\ Error, not checked!")
    print(jacs)
    print(times)


def testPseudoTree(glueByVertex=True):
    cycle = Graph.cycle(5)
    cycle.setEdgeState(0, 1, 1)
    cycle.setEdgeState(1, 2, 1)
    cycle.setEdgeState(2, 3, 1)
    cycle.setEdgeState(3, 4, 2)
    # cycle.visualize()
    audit = cycle.auditEdges()
    print(f"Cycle sources: {audit[0]}, Cycle sinks: {audit[1]}")
    print(f"Cycle Picard: {prettyCok(cycle.pic())}")
    adjacency = [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ]
    stick = Graph(adjacency)
    audit = stick.auditEdges()
    print(f"Tree sources: {audit[0]}, Tree sinks: {audit[1]}")
    print(f"Tree picard: {prettyCok(stick.pic())}")
    gluePoint = 1
    if glueByVertex:
        glued = Graph.glueByVertex(cycle, stick, gluePoint, 0)
    else:
        glued = Graph.glueByEdge(cycle, stick, gluePoint, 0, state=2)
    glued.visualize()
    audit = glued.auditEdges()
    print(f"Pseudo-tree sources: {audit[0]}, Pseudo-tree sinks: {audit[1]}")
    print(f"Pseudo-tree picard: {prettyCok(glued.pic())}")


def cycleOrientation(size: int):
    cycle = Graph.cycle(size)
    cycle.setEdgeState(0, 1, 1)
    cycle.setEdgeState(1, 2, 2)
    cycle.setEdgeState(2, 3, 1)
    # cycle.setEdgeState(3, 4, 2)
    # cycle.setEdgeState(4, 5, 1)
    """cycle.setEdgeState(5, 6, 0)
    cycle.setEdgeState(6, 7, 0)
    cycle.setEdgeState(7, 8, 0)
    cycle.setEdgeState(8, 9, 0)
    cycle.setEdgeState(9, 10, 0)"""
    cycle.setEdgeState(size - 1, 0, 2)
    cycle.visualize()
    print(f"Cycle Picard of size {size}: {prettyCok(cycle.pic())}")
    print(f"Cycle paths: {cycle.countPaths()}")


def wheelOrientation(size: int):
    wheel = Graph.wheel(size, direction=0, spokeDirection=0)
    # wheel.setEdgeState(1, 2, 1)
    # wheel.setEdgeState(2, 3, 1)
    # wheel.setEdgeState(3, 4, 1)
    # wheel.setEdgeState(4, 5, 1)
    """wheel.setEdgeState(5, 6, 0)
    wheel.setEdgeState(6, 7, 0)
    wheel.setEdgeState(7, 8, 0)
    wheel.setEdgeState(8, 9, 0)
    wheel.setEdgeState(9, 10, 0)"""
    # wheel.setEdgeState(size - 1, 1, 1)
    wheel.visualize()
    print(f"Wheel Picard of size {size}: {prettyCok(wheel.pic())}")


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
    # cokDict = {i: allStats(Graph.cycle(i), skipRotations=False) for i in range(8, 9)}
    # logger.info(cokDict)
    # testBruteForce((3, 14))
    # graph = Graph(adjacency)
    # graph.visualize()
    # print(graph.auditEdges())
    # testPseudoTree(glueByVertex=True)
    # testAllJacs()
    """Graph.glueByEdge(Graph.cycle(5), Graph([[0, 1, 1], [1, 0, 0], [1, 0, 0]]), vertex1=4, vertex2=0).visualize(
        title="A Pseudo-Tree Graph", positions={0: [-.5, 0], 4: [.5, 0], 1: [-.9, .5], 3: [.9, .5], 5: [0, -.5]}
    )"""
    [plotFactors('cycle', (size, 8), includePaths=(2,), bySize=False) for size in range(4, 11)]
    # [wheelOrientation(i) for i in range(3, 21)]
    # cycleOrientation(4)
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
    print(f"Tree picard: {prettyCok(stick.pic())}")"""
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
    print(prettyCok(Utils.coKernel(graph.laplacian)))"""
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

