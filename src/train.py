import random
import numpy as np
import math
import Utils


class Trainer:
    """Abstract parent class representing a training method for the neural network"""
    def prime(self, population, topography, loss):
        """Initialize additional properties"""
        self.fittneses = [0]
        self.popSize = 1
        self.population = population
        self.loss = loss
        self.topography = topography
        self.type = ""

    def selection(self, selectCount):
        """Selects the best `selectCount` number of networks from the population.
        Selects the first if there is only one in the population"""
        if len(self.population["pop"]) == 1:
            return [0]
        # Roulette wheel selection for multiple networks in population
        selectedIds = []
        totalFitness = sum(self.fittneses)
        while len(selectedIds) < selectCount:
            randPlace = random.random()
            rouletteSum = 0
            current = -1
            # Add to sum while greater than pointer
            while randPlace > rouletteSum:
                current += 1
                rouletteSum += self.fittneses[current]/totalFitness
            if current not in selectedIds:
                # Append to selected ids
                selectedIds.append(current)
        return selectedIds

    def getFitness(self, memberId, samples):
        """Returns the fittness of the given network over the given sample set"""
        return 1 / self.loss(samples, memberId)[0]

    def setAllFitness(self, samples):
        """Sets the fittness array to an array of fittnesses of the elements"""
        for i in range(0, self.popSize):
            self.fittneses[i] = self.getFitness(i, samples)

    def train(self, sampleList, classList, epochs: int = 1000, displayUpdate: int = 10, verbosity: int = 0, showPlots: bool = False):
        """Trains the population of networks over the given sample set.  'Generic' method, functionality is not
        specific to one trainer algorithm"""
        epochLosses = []
        epochAccuracy = []
        epochTimes = []
        # Append column to 1's to allow for training thresholds
        sampleList = np.c_[sampleList, np.ones((sampleList.shape[0]))]
        sampleList = np.atleast_2d(sampleList)
        classList = np.atleast_2d(classList)
        for epoch in range(0, epochs):
            # Train on each sample
            sampleNum = 0
            for (sample, classOf) in zip(sampleList, classList):
                self.epochTrain(sample, classOf)
                sampleNum += 1
            # Check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                totalLoss = [0, 0]
                bestLoss = [float("infinity"), 0]
                mostCorrect = [0, 0]
                for i in range(0, len(self.population["pop"])):
                    loss = self.loss([sampleList, classList], i, verbosity)
                    totalLoss[0] += loss[0]
                    totalLoss[1] += loss[1]
                    if loss[0] < bestLoss[0]:
                        bestLoss = loss
                    if loss[1] > mostCorrect[1]:
                        mostCorrect = loss
                totalLoss[0] /= len(self.population["pop"])
                totalLoss[1] /= len(self.population["pop"])
                # Display network with the lowest loss and accuracy and the overall loss and accuracy for the population
                epochLosses.append(totalLoss[0])
                epochAccuracy.append(totalLoss[1])
                epochTimes.append(epoch)
                print("Epoch: " + str(epoch) + ", Average Loss: " + str(totalLoss[0]) + (
                            ", Correct: " + str(totalLoss[1] * 100) + "%" if verbosity > 0 else ""), end="")
                if len(self.population["pop"]) > 1:
                    print(", Best loss: "+str(bestLoss)+", most correct: "+str(mostCorrect))
                else:
                    print()
        # Show python plots
        if showPlots:
            algorithmName = "Backpropagation" if self.type == "backprop" else "Genetic Algorithm"
            Utils.plot(epochTimes, epochLosses, "Loss", algorithmName+" Loss Over Epochs")
            Utils.plot(epochTimes, epochAccuracy, "Accuracy", algorithmName+" Accuracy Over Epochs")

    def epochTrain(self, sample, classOf):
        """Stub method to be overriden by subclasses.  This method is different for each trainer for their own
        specific needs"""
        pass

    def initMember(self):
        """Initialize network and add to the population"""
        self.population["pop"].append([])
        # Initialize weights and thresholds for all but last layer
        for i in range(0, len(self.topography) - 2):
            weight = np.random.randn(self.topography[i] + 1, self.topography[i + 1] + 1)
            self.population["pop"][-1].append(weight / np.sqrt(self.topography[i]))
        # Initialize weights and thresholds for output layer
        weight = np.random.randn(self.topography[-2] + 1, self.topography[-1])
        self.population["pop"][-1].append(weight / np.sqrt(self.topography[-2]))


class Backpropogator(Trainer):
    """Trainer method for backpropogation"""
    def __init__(self, learningRate):
        super().__init__()
        self.learningRate = learningRate

    def prime(self, population, topography, loss):
        """Initialize this network"""
        super().prime(population, topography, loss)
        self.type = "backprop"
        super().initMember()

    def epochTrain(self, sample, classOf):
        """Train this network for one epoch"""
        activations = [np.asarray([sample])]
        # Gather activations for each layer
        for layer in range(0, len(self.population["pop"][0])):
            activation = activations[layer].dot(self.population["pop"][0][layer])
            activations.append(Utils.sigmoid(activation))
        # Calculate error of output layer
        error = activations[-1] - classOf
        deltas = [error * Utils.sigmoidDx(activations[-1])]
        # Calculate change in weights
        for layer in range(len(activations) - 2, 0, -1):
            delta = deltas[-1].dot(self.population["pop"][0][layer].transpose())
            delta = delta * Utils.sigmoidDx(activations[layer])
            deltas.append(delta)
        # Update weights
        for layer in range(0, len(self.population["pop"][0])):
            self.population["pop"][0][layer] += -self.learningRate * activations[layer].transpose().dot(deltas[-(layer + 1)])
