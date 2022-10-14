import math
import numpy as np
import csv
import matplotlib.pyplot as plt


def sigmoid(x):
    """Sigmoid function"""
    return 1.0 / (1 + np.exp(-x))


def sigmoidDx(x):
    """Derivative of sigmoid where `x` is already the output of the sigmoid function"""
    # sig'(x) = sig(x) * (1 - sig(x))
    return x * (1 - x)


def tanh(x):
    """Calculates the tanh function"""
    try:
        math.pow(math.e, -0.667 * x)
    except Exception as e:
        print(e)
    return 3.432/(1+math.pow(math.e, -0.667*x)) - 1.716


def importCSV(fileName, normDict, classifier, trainTestSplit=0.8, normalize=True):
    """Imports the data from a CSV file, normalize the data and assigns classes to each sample point"""
    samples = []
    classes = []
    with open(fileName, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rowList = []
            # Normalize data and add to list
            for key, value in normDict.items():
                rowList.append(float(row[key])/(value[0] if normalize else 1))
            classList = [0, 0]
            classList[int(row[classifier])] = 1
            classes.append(classList)
            samples.append(rowList)
    totalAttribs = np.asarray(samples)
    totalClass = np.asarray(classes)
    # Training set, a tuple of attribute and class data
    trainSet = ([], [])
    # Testing set, a tuple of attribute and class data
    testSet = ([], [])
    i = 0
    # Divide sample points into training and testing sets
    while i < len(totalAttribs):
        if i < trainTestSplit * len(totalAttribs):
            trainSet[0].append(totalAttribs[i])
            trainSet[1].append(totalClass[i])
        else:
            testSet[0].append(totalAttribs[i])
            testSet[1].append(totalClass[i])
        i += 1
    return trainSet, testSet


def plot(xPoints, yPoints, yLabel, title):
    """Creates a pyplot scatter plot for the given points"""
    plt.scatter(xPoints, yPoints)
    plt.plot(xPoints, yPoints)
    plt.xlabel("Epoch")
    plt.ylabel(yLabel)
    plt.title(title)
    plt.show()




