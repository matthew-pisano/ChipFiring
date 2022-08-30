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
    cokDict = {i: bruteCheckGraphs(Graph.cycle(i)) for i in range(*rng)}
    logger.info(cokDict)


def testSinkSource(graph: Graph):
    print(prettyCok(graph.pic()))
    graph.forceFlow(0, makeSink=True)
    print(prettyCok(graph.pic()))
    graph.forceFlow(0, makeSink=False)
    print(prettyCok(graph.pic()))


def testPseudoTree(glueByVertex=True):
    cycle = Graph.cycle(8)
    cycle.setEdgeState(3, 4, 1)
    cycle.setEdgeState(2, 3, 2)
    cycle.setEdgeState(1, 2, 2)
    cycle.setEdgeState(0, 1, 2)
    cycle.setEdgeState(7, 6, 1)
    cycle.setEdgeState(0, 7, 2)
    cycle.setEdgeState(5, 6, 1)
    cycle.setEdgeState(4, 5, 2)
    # cycle.visualize()
    audit = cycle.auditEdges()
    print(f"Cycle sources: {audit[0]}, Cycle sinks: {audit[1]}")
    print(f"Cycle Picard: {prettyCok(cycle.pic())}")
    adjacency = [
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
    ]
    stick = Graph(adjacency)
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
    """adjacency = [
        [0, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    stick = Graph(adjacency)
    stick.visualize()
    audit = stick.auditEdges()
    print(f"Tree sources: {audit[0]}, Tree sinks: {audit[1]}")
    print(f"Tree picard: {prettyCok(stick.pic())}")"""
    testBruteForce((17, 20))
    """graph = Graph.complete(3)
    testSinkSource(graph)"""
    """print(Utils.coKernel(
        np.copy(
            [[1, -1, 0, 0],
            [0, 0, 0, 0],
             [-1, 0, 1, 0],
             [0, -1, -1, 2]]
        )
    ))
    print(Utils.coKernel(
        np.copy(
            [[1, -1, 0, 0],
             [0, 1, 0, -1],
             [-1, 0, 2, -1],
             [0, 0, 0, 0]]
        )
    ))
    print(Utils.coKernel(
        np.copy(
            [[2, -1, -1, 0],
             [0, 1, 0, -1],
             [0, 0, 1, -1],
             [0, 0, 0, 0]]
        )
    ))"""
    """print(prettyCok(Utils.coKernel(
        np.copy(
            [[0, 0, 0, 0],
             [-1, 2, -1, 0],
             [0, 0, 1, -1],
             [0, 0, 0, 0]]
        )
    )))
    print(prettyCok(Utils.coKernel(
        np.copy(
            [[0, 0, 0, 0, 0],
             [-1, 2, -1, 0, 0],
             [0, 0, 1, -1, 0],
             [0, 0, 0, 1, -1],
             [0, 0, 0, 0, 0]]
        )
    )))
    print(prettyCok(Utils.coKernel(
        np.copy(
            [[0, 0, 0, 0, 0],
             [-1, 2, -1, 0, 0],
             [0, 0, 1, -1, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, -1, 1]]
        )
    )))
    print(prettyCok(Utils.coKernel(
        np.copy(
            [[0, 0, 0],
             [-1, 2, -1],
             [0, 0, 0]]
        )
    )))"""
    # graph.visualize()
        # print(f"Graph {idx}", graph.jac())
        # current.visualize()
    # graph.visualize()
    # cyclediv = fireCycle(graph, divisor)
    # print(cyclediv)
    # grediv = greedy(graph, divisor)
    # print(grediv)
    # graph.visualize(grediv[2])
    # qdiv = qReduced(graph, divisor)
    # print(qdiv)
    # graph.visualize(qdiv[2])

    # print(graph.jac(divisor))
    # print(graph.pic(divisor))
    """for v in range(len(graph)):
        print(f"Jacobian at vertex {v}:", graph.jac(vertex=v))
    print("Picard Group:", graph.pic())"""
    end = time.time()
    logger.info(f"=====\nExiting program at {datetime.datetime.fromtimestamp(end).strftime('%H:%M:%S')}"
          f"\nElapsed time: {round(end-start, 3)}s\n=====")

