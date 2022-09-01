import logging
import os.path
import time
import typing

import numpy as np
import sys

DEBUG = {"debug_out": False, "log_file": "loggy.log", "snfDepth": None}


loglevel = logging.INFO
if DEBUG["debug_out"]:
    loglevel = logging.DEBUG

if DEBUG["log_file"]:
    filePath = "src/logs/"+DEBUG["log_file"]
    with open(filePath, "w") as file:
        file.write("===> Log File <===\n")
    fileHandler = logging.FileHandler(filePath)
    shell = logging.StreamHandler(sys.stdout)
    logging.basicConfig(handlers=[shell, fileHandler], level=loglevel,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

logger = logging.getLogger(__name__)


class Utils:
    @classmethod
    def smithNormalForm(cls, matrix: np.ndarray):
        """Calculates the smith normal form of the given matrix"""
        start = time.time()
        smith = matrix.copy()

        def exchangeRows(other: np.ndarray, i: int, j: int):
            smith[[i, j]] = smith[[j, i]]
            other[[i, j]] = other[[j, i]]

        def exchangeCols(other: np.ndarray, i: int, j: int):
            smith[:, [i, j]] = smith[:, [j, i]]
            other[:, [i, j]] = other[:, [j, i]]

        def addRows(other: np.ndarray, i: int, j: int, scale=1):
            smith[i, :] = (smith[i, :] + scale * smith[j, :])
            other[i, :] = (other[i, :] + scale * other[j, :])

        def addCols(other: np.ndarray, i: int, j: int, scale=1):
            smith[:, i] = (smith[:, i] + scale * smith[:, j])
            other[:, i] = (other[:, i] + scale * other[:, j])

        def scaleRow(other: np.ndarray, i: int, scale):
            smith[i, :] = scale * smith[i, :]
            other[i, :] = scale * other[i, :]

        def scaleCol(other: np.ndarray, i: int, scale=1):
            smith[:, i] = scale * smith[:, i]
            other[:, i] = scale * other[:, i]

        matrixSize = len(smith) if not DEBUG["snfDepth"] else DEBUG["snfDepth"]

        def minAij(s):
            """Find the minimum non-zero element below and to the right of matrix[s][s]"""
            element = [s, s]
            # globalMin = max(max([abs(x) for x in smith[j][s:]]) for j in range(s, matrixSize))
            globalMin = float('inf')
            for i in (range(s, matrixSize)):
                for j in (range(s, matrixSize)):
                    if smith[i][j] != 0 and abs(smith[i][j]) <= globalMin:
                        element = [i, j]
                        globalMin = abs(smith[i][j])
            return element

        def isLone(s):
            """Checks if matrix[s][s] is the only non-zero in col s below matrix[s][s] and the only
            non-zero in row s to the right of matrix[s][s]"""
            if [smith[s][x] for x in range(s + 1, matrixSize) if smith[s][x] != 0] + [smith[y][s]
                    for y in range(s + 1, matrixSize) if smith[y][s] != 0] == []:
                return True
            else:
                return False

        def findNonDivisible(s):
            """Finds the first element which is not divisible by matrix[s][s]"""
            for x in range(s + 1, matrixSize):
                for y in range(s + 1, matrixSize):
                    if smith[x][y] % smith[s][s] != 0:
                        return x, y
            return None

        p = np.identity(matrixSize)
        q = np.identity(matrixSize)
        for s in range(0, matrixSize):
            while not isLone(s):
                # Get min location
                i, j = minAij(s)
                exchangeRows(p, s, i)
                exchangeCols(q, s, j)
                for x in range(s + 1, matrixSize):
                    if smith[x][s] != 0:
                        k = smith[x][s] // smith[s][s]
                        addRows(p, x, s, -k)
                for x in range(s + 1, matrixSize):
                    if smith[s][x] != 0:
                        k = smith[s][x] // smith[s][s]
                        addCols(q, x, s, -k)
                if isLone(s):
                    res = findNonDivisible(s)
                    if res:
                        x, y = res
                        addRows(p, s, x, 1)
                    else:
                        if smith[s][s] < 0:
                            scaleRow(p, s, -1)
            if smith[s][s] < 0:
                scaleRow(p, s, -1)

        # logger.info(f"Smith time for size {len(matrix)} is {time.time()-start}s")
        return smith, p, q

    @classmethod
    def coKernel(cls, matrix: np.ndarray, divisor: typing.Any | np.ndarray = None):
        """Returns the polynomial, invariant factors, and rank of the coKernel of the given matrix"""
        smith, p, q = cls.smithNormalForm(matrix)
        # print(f"{len(matrix)} Reduced:", matrix)
        # print(f"{len(matrix)} Smith:", smith)
        if not divisor:
            product = p
        else:
            product = np.matmul(p, divisor)
        infs = []
        invFactors: list[int] = []
        delModifier = 0
        for i in range(len(smith)):
            if smith[i][i] == 1 or np.all((smith[i] == 0)):
                product = np.delete(product, i + delModifier, axis=0)
                delModifier -= 1
                if np.all((smith[i] == 0)):
                    infs.append(float("inf"))
            else:
                invFactors.append(int(smith[i][i]))
        product = [np.atleast_1d(layer).tolist() for layer in product]
        if len(infs) > 0:
            product.append(infs)

        # polynomial, invariant factors, rank
        return product, invFactors, len(infs)
