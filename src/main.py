import datetime

from algorithms import *
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
    cycle = Graph.cycle(7)
    cycle.setEdgeState(3, 4, 2)
    cycle.setEdgeState(2, 3, 1)
    cycle.setEdgeState(1, 2, 2)
    cycle.setEdgeState(0, 1, 1)
    cycle.setEdgeState(5, 6, 2)
    cycle.setEdgeState(4, 5, 1)
    print(cycle.pic())
    adjacency = [
        [0, 0, 0],
        [1, 0, 1],
        [0, 0, 0],
    ]
    stick = Graph(adjacency)
    print(stick.pic())
    glued = Graph.glue(cycle, stick, 3, 0)
    glued.visualize()
    print(glued.auditEdges())
    print(glued.pic())
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

