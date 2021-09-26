import csv
import multiprocessing
import neat
import neat.visualize as vs
import numpy as np
import os
import pickle
import scikitplot
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score

if __name__=='__main__':
    x = pickle.load(open('../../../../datasets/x-no-2', 'rb'))
    y = pickle.load(open('../../../../datasets/y-no-2', 'rb'))

    for i in range(len(y)):
        if(y[i] == 3):
            y[i] = 1
        if(y[i] == 5):
            y[i] = 2
        if(y[i] == 6):
            y[i] = 3

    accuracies_train = []
    accuracies_test = []
    precisions = []
    recalls = []
    f1_scores = []
    kf = StratifiedKFold(n_splits=10)
    n_fold = 1

    for train_index, test_index in kf.split(x, y):
        def eval_genomes(genome, config):
            net = neat.nn.RecurrentNetwork.create(genome, config)
            outputs = []

            for xi in x[train_index]:
                outputs.append(np.argmax(net.activate(xi)))
            
            return accuracy_score(y[train_index], outputs)

        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                'config.txt')
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        pe = neat.ParallelEvaluator(5, eval_genomes)
        winner = p.run(pe.evaluate, 1500)
        
        winner_net = neat.nn.RecurrentNetwork.create(winner, config)

        outputs_train = []
        for xi in x[train_index]:
            outputs_train.append(np.argmax(winner_net.activate(xi)))           
        
        accuracies_train.append(accuracy_score(y[train_index], outputs_train))

        outputs_test = []
        for xi in x[test_index]:
            outputs_test.append(np.argmax(winner_net.activate(xi)))
        
        scikitplot.metrics.plot_confusion_matrix(y[test_index], outputs_test, title="Four Classes Confusion Matrix: Fold " + str(n_fold) , normalize=True, text_fontsize="small")
        plt.savefig("4-classes-cf-" + str(n_fold) + ".png")
        plt.clf()
        plt.close()

        vs.draw_net(config, winner, False, filename='4-classes-winner-' + str(n_fold), show_disabled=False)
        with open('4-classes-winner-' + str(n_fold), 'wb') as f:
            pickle.dump(winner, f)
        
        with open('4-classes-results-' + str(n_fold), 'w') as f:
            f.write('\nBest genome:\n{!s}'.format(winner))
        
        accuracies_test.append(accuracy_score(y[test_index], outputs_test))
        precisions.append(precision_score(y[test_index], outputs_test, average='macro', zero_division=0))
        recalls.append(recall_score(y[test_index], outputs_test, average='macro', zero_division=0))
        f1_scores.append(f1_score(y[test_index], outputs_test, average='macro', zero_division=0))
        
        n_fold += 1

    with open("results-4-classes.csv", 'w', encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        fold = 1
        header = ['fold', 'accuracy train', 'accuracy test', 'precision', 'recall', 'f1_score']
        writer.writerow(header)
        for atrain, atest, p, r, f1 in zip(accuracies_train,
                                            accuracies_test, 
                                            precisions,
                                            recalls,
                                            f1_scores):
            row = [fold, atrain, atest, p, r, f1]
            writer.writerow(row)
            fold += 1

        avgs = ["Average",
                np.mean(np.array(accuracies_train)),
                np.mean(np.array(accuracies_test)),
                np.mean(np.array(precisions)),
                np.mean(np.array(recalls)),
                np.mean(np.array(f1_scores)) ]
        writer.writerow(avgs)