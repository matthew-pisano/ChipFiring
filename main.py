from algorithms import *


if __name__ == "__main__":
    adjacency = [
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 1, 1, 1],
    ]
    divisor = Divisor([1, -6, 8, -1])
    graph = Graph(adjacency)
    print(qReduced(graph, divisor, mutate=True))
    graph.visualize(divisor)
    """result = greedy(graph, divisor)
    print(f'Is winnable: {result[0]}')
    if result[0]:
        graph.visualize(result[2])"""
    # print(dhar(graph, divisor, 0))
