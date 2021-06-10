# Experiment with tanh relu and sigmoid activation functions

import os
import neat
import pickle
import numpy as np
import sys
import multiprocessing

# Extract data from binaries
x_train_data = pickle.load(open("../../nn-experiments/X_train.pickle", "rb"))
x_test_data = pickle.load(open("../../nn-experiments/X_test.pickle", "rb"))
y_train_data = pickle.load(open("../../nn-experiments/y_train.pickle", "rb"))
y_test_data = pickle.load(open("../../nn-experiments/y_test.pickle", "rb"))

X = []
y = []
X_test = []
y_test = []

for x in x_train_data:
    X.append(x)

for x in x_test_data:
    X_test.append(x)

for yo in y_train_data.tolist():
    if yo == 0:
        y.append([1, 0, 0, 0, 0, 0, 0])
    if yo == 1:
        y.append([0, 1, 0, 0, 0, 0, 0])
    if yo == 2:
        y.append([0, 0, 1, 0, 0, 0, 0])
    if yo == 3:
        y.append([0, 0, 0, 1, 0, 0, 0])
    if yo == 4:
        y.append([0, 0, 0, 0, 1, 0, 0])
    if yo == 5:
        y.append([0, 0, 0, 0, 0, 1, 0])
    if yo == 6:
        y.append([0, 0, 0, 0, 0, 0, 1])

for yo in y_test_data.tolist():
    if yo == 0:
        y_test.append([1, 0, 0, 0, 0, 0, 0])
    if yo == 1:
        y_test.append([0, 1, 0, 0, 0, 0, 0])
    if yo == 2:
        y_test.append([0, 0, 1, 0, 0, 0, 0])
    if yo == 3:
        y_test.append([0, 0, 0, 1, 0, 0, 0])
    if yo == 4:
        y_test.append([0, 0, 0, 0, 1, 0, 0])
    if yo == 5:
        y_test.append([0, 0, 0, 0, 0, 1, 0])
    if yo == 6:
        y_test.append([0, 0, 0, 0, 0, 0, 1])

X = np.asarray(X)
y = np.asarray(y)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

def eval_genomes(genome, config):
    fitness = 0.0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    for xi, yo in zip(X, y):
        output = net.activate(xi)
        if (np.argmax(yo) == np.argmax(output)):
            fitness += 1.0
    fitness /= len(X)
    return fitness

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    pe = neat.ParallelEvaluator(2, eval_genomes)
    winner = p.run(pe.evaluate, 2000)
    
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    accuracy = 0.0

    for xi, yo in zip(X_test, y_test):
        output = winner_net.activate(xi)
        if (np.argmax(yo) == np.argmax(output)):
            accuracy += 1
    
    print("\nAccuracy: ", accuracy/len(y))
    file1 = open("results-05d.txt", "w")
    file1.write("Accuracy test: ")
    file1.write(str(accuracy/len(y)))
    file1.write('\nBest genome:\n{!s}'.format(winner))
    file1.close()
    
    with open('winner-5d', 'wb') as f:
        pickle.dump(winner, f)


if __name__ == '__main__':
    run('config-feedforward.txt')