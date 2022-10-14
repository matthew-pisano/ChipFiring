import os
import time
from NeuralNet import NeuralNet
from train import *


def testProject(netConfig, trainingConfig):
    """Runs any supported configuration of the entire project"""
    # The number of nodes in each layer of the neural network
    netShape = [2, 2, 1]
    # Trainer passed into NeuralNet object determines what kind of algorithm will be used to train it
    net = NeuralNet(netShape, Backpropogator(netConfig["learningRate"]))
    t = time.time()

    trainSet = [[[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [1]]]
    testSet = [[[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [1]]]

    # Train the neural network
    net.train(np.array(trainSet[0]), np.array(trainSet[1]), **trainingConfig)
    timeDiff = time.time() - t
    print("Trained after " + str(timeDiff) + "s")
    print("================================\n\n==============================")
    # Calculate loss of the trained neural network
    sampleSet = np.c_[np.array(testSet[0]), np.ones((np.array(testSet[0]).shape[0]))]
    loss = net.loss([sampleSet, np.array(testSet[1])], verbosity=3)
    # Print out results
    print("Loss: " + str(loss[0]) + ", Correct: " + str(loss[1] * 100) + "%")
    print("Overall score: " + str(loss[1] / timeDiff))


if __name__ == "__main__":
    backpropConfig = {
        "learningRate": 0.18

    }
    trainingConfig = {
        "epochs": 700,
        "displayUpdate": 1,
        "verbosity": 1,
        "showPlots": True
    }
    testProject(backpropConfig, trainingConfig)
