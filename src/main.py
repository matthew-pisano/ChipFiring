import datetime

from algorithms import *
from src import results
from utils import logger


def testSrcSnk():
    graph = Graph.cycle(4)
    graph.forceFlow(0)
    graph.visualize()
    print(graph.jac(2))
    graph.forceFlow(0, makeSink=False)
    graph.visualize()
    print(graph.jac(2))


def testBruteForce(rng=(3, 10)):
    cokDict = {i: bruteCheckGraphs(Graph.cycle(i), includeBi=False) for i in range(*rng)}
    logger.info(cokDict)


def testSinkSource(graph: Graph):
    print(prettyCok(graph.pic()))
    graph.forceFlow(0, makeSink=True)
    print(prettyCok(graph.pic()))
    graph.forceFlow(0, makeSink=False)
    print(prettyCok(graph.pic()))


def testPseudoTree(glueByVertex=True):
    cycle = Graph.cycle(6)
    cycle.setEdgeState(3, 4, 1)
    cycle.setEdgeState(5, 0, 1)
    cycle.setEdgeState(2, 3, 2)
    cycle.setEdgeState(1, 2, 2)
    cycle.setEdgeState(0, 1, 2)
    cycle.setEdgeState(4, 5, 2)
    cycle.visualize()
    audit = cycle.auditEdges()
    print(f"Cycle sources: {audit[0]}, Cycle sinks: {audit[1]}")
    print(f"Cycle Picard: {prettyCok(cycle.pic())}")
    adjacency = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
    ]
    stick = Graph(adjacency)
    stick.addEdge(0, 3, state=2)
    audit = stick.auditEdges()
    print(f"Tree sources: {audit[0]}, Tree sinks: {audit[1]}")
    print(f"Tree picard: {prettyCok(stick.pic())}")
    if glueByVertex:
        glued = Graph.glueByVertex(cycle, stick, 3, 0)
    else:
        glued = Graph.glueByEdge(cycle, stick, 3, 0, state=2)
    glued.visualize()
    audit = glued.auditEdges()
    print(f"Pseudo-tree sources: {audit[0]}, Pseudo-tree sinks: {audit[1]}")
    print(f"Pseudo-tree picard: {prettyCok(glued.pic())}")


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
        [0, 1, 1],
        [0, 0, 1],
        [1, 0, 0],
    ]
    divisor = Divisor([16, -4, -5, 0])
    graph = Graph(adjacency)"""
    # testBruteForce((3, 14))
    # graph = Graph(adjacency)
    # graph.visualize()
    # print(graph.auditEdges())
    # testPseudoTree(glueByVertex=False)
    adjacency = [
        [0, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    stick = Graph(adjacency)
    # stick.visualize()
    audit = stick.auditEdges()
    print(f"Tree sources: {audit[0]}, Tree sinks: {audit[1]}")
    print(f"Tree picard: {prettyCok(stick.pic())}")
    # testBruteForce((3, 15))
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
    """graph = Graph.cycle(6)
    graph.setEdgeState(0, 1, state=2)
    graph.setEdgeState(1, 2, state=1)
    graph.setEdgeState(2, 3, state=2)
    graph.setEdgeState(3, 4, state=1)
    graph.setEdgeState(4, 5, state=2)
    graph.setEdgeState(5, 0, state=1)
    graph.visualize()
    print(f"Graph picard: {prettyCok(graph.pic())}")"""

    # print(graph.jac(divisor))
    # print(graph.pic(divisor))
    """for v in range(len(graph)):
        print(f"Jacobian at vertex {v}:", graph.jac(vertex=v))
    print("Picard Group:", graph.pic())"""
    end = time.time()
    logger.info(f"=====\nExiting program at {datetime.datetime.fromtimestamp(end).strftime('%H:%M:%S')}"
          f"\nElapsed time: {round(end-start, 3)}s\n=====")

