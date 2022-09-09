from algorithms import *


def testPseudoTree(glueByVertex=True):
    """Results in a cycle and pseudo-tree Jacobian that is not equal to Z"""
    cycle = Graph.cycle(7)
    cycle.setEdgeState(3, 4, 2)
    cycle.setEdgeState(2, 3, 1)
    cycle.setEdgeState(1, 2, 2)
    cycle.setEdgeState(0, 1, 1)
    cycle.setEdgeState(0, 6, 2)
    cycle.setEdgeState(5, 6, 2)
    cycle.setEdgeState(4, 5, 1)
    # cycle.visualize()
    audit = cycle.auditEdges()
    print(f"Cycle sources: {audit[0]}, Cycle sinks: {audit[1]}")
    print(f"Cycle Picard: {prettyCok(cycle.pic())}")
    adjacency = [
        [0, 0, 0],
        [1, 0, 1],
        [0, 0, 0],
    ]
    stick = Graph(adjacency)
    audit = stick.auditEdges()
    print(f"Tree sources: {audit[0]}, Tree sinks: {audit[1]}")
    print(f"Tree picard: {prettyCok(stick.pic())}")
    if glueByVertex:
        glued = Graph.glueByVertex(cycle, stick, 3, 0)
    else:
        glued = Graph.glueByEdge(cycle, stick, 3, 0)
    # glued.visualize()
    audit = glued.auditEdges()
    print(f"Pseudo-tree sources: {audit[0]}, Pseudo-tree sinks: {audit[1]}")
    print(f"Pseudo-tree picard: {prettyCok(glued.pic())}")


def testPseudoTree(glueByVertex=True):
    """Sinks i glued are not rank(pic)"""
    cycle = Graph.cycle(6)
    # cycle.setEdgeState(3, 4, 1)
    cycle.setEdgeState(4, 3, 1)
    cycle.setEdgeState(2, 3, 1)
    cycle.setEdgeState(1, 2, 2)
    cycle.setEdgeState(0, 1, 1)
    # cycle.setEdgeState(4, 5, 2)
    cycle.visualize()
    audit = cycle.auditEdges()
    print(f"Cycle sources: {audit[0]}, Cycle sinks: {audit[1]}")
    print(f"Cycle Picard: {prettyCok(cycle.pic())}")
    adjacency = [[0]]
    stick = Graph(adjacency)
    audit = stick.auditEdges()
    print(f"Tree sources: {audit[0]}, Tree sinks: {audit[1]}")
    print(f"Tree picard: {prettyCok(stick.pic())}")
    gluePoint = 1
    if glueByVertex:
        glued = Graph.glueByVertex(cycle, stick, gluePoint, 0)
    else:
        glued = Graph.glueByEdge(cycle, stick, gluePoint, 0, state=0)
    glued.visualize()
    audit = glued.auditEdges()
    print(f"Pseudo-tree sources: {audit[0]}, Pseudo-tree sinks: {audit[1]}")
    print(f"Pseudo-tree picard: {prettyCok(glued.pic())}")


def testPseudoTree(glueByVertex=True):
    """When gluing by vertex, picard is not Z_3 x Z^3.  This does not match
    the edge glued version"""
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
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 0]
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
