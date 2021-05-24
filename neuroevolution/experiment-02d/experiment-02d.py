# Experiments with tanh and sigmoid activation functions

import os
import neat
import pickle
import numpy as np
import sys

sys.path.append('../../libs')

# Extract data from binaries
x_train_data = pickle.load(open("../../nn-experiments/X_train.pickle", "rb"))
x_test_data = pickle.load(open("../../nn-experiments/X_test.pickle", "rb"))
y_train_data = pickle.load(open("../../nn-experiments/y_train.pickle", "rb"))
y_test_data = pickle.load(open("../../nn-experiments/y_test.pickle", "rb"))

y_data = np.append(y_train_data, y_test_data)

X = []
y = []

for x in x_train_data:
    X.append(x)

for x in x_test_data:
    X.append(x)

for yo in y_data.tolist():
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

X = np.asarray(X)
y = np.asarray(y)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, yo in zip(X, y):
            output = net.activate(xi)
            if (np.argmax(yo) == np.argmax(output)):
                genome.fitness += 1.0
        genome.fitness /= len(X)

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 1000)
    
    file1 = open("results.txt", "w")
    file1.write('\nBest genome:\n{!s}'.format(winner))
    file1.close()
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    accuracy = 0.0
    for xi, yo in zip(X, y):
        output = winner_net.activate(xi)
        if (np.argmax(yo) == np.argmax(output)):
            accuracy += 1
    print("\nAccuracy: ", accuracy/len(y))

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedfordward.txt')
run(config_path)