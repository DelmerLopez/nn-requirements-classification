import csv
import multiprocessing
import neat
import numpy as np
import pickle
from matplotlib import pyplot as plt
import scikitplot
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

if __name__=='__main__':
    x = pickle.load(open('../../datasets/x-no-4', 'rb'))
    y = pickle.load(open('../../datasets/y-no-4', 'rb'))

    Y = []
    for o in y:
        if o == 0:
            Y.append([1, 0, 0, 0, 0, 0, 0])
        if o == 1:
            Y.append([0, 1, 0, 0, 0, 0, 0])
        if o == 2:
            Y.append([0, 0, 1, 0, 0, 0, 0])
        if o == 3:
            Y.append([0, 0, 0, 1, 0, 0, 0])
        if o == 4:
            Y.append([0, 0, 0, 0, 1, 0, 0])
        if o == 5:
            Y.append([0, 0, 0, 0, 0, 1, 0])
        if o == 6:
            Y.append([0, 0, 0, 0, 0, 0, 1])

    accuracies_train = []
    accuracies_test = []
    precisions = []
    recalls = []
    f1_scores = []
    kf = StratifiedKFold(n_splits=10)
    n_fold = 1

    for train_index, test_index in kf.split(x, y):

        def eval_genomes(genome, config):
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                outputs = []

                for xi in x[train_index]:
                    outputs.append(np.argmax(net.activate(xi)))
                
                return precision_score(y[train_index], outputs, average='micro', zero_division=0)
        
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                'config-feedforward-neuroevolution.txt')
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genomes)
        winner = p.run(pe.evaluate, 1000)
        
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

        outputs_train = []
        for xi in x[train_index]:
            outputs_train.append(np.argmax(winner_net.activate(xi)))
        
        accuracies_train.append(accuracy_score(y[train_index], outputs_train))

        outputs_test = []
        for xi in x[test_index]:
            outputs_test.append(np.argmax(winner_net.activate(xi)))

        scikitplot.metrics.plot_confusion_matrix(y[test_index], outputs_test, title="Five Classes Confusion Matrix: Fold " + str(n_fold) , normalize=True, text_fontsize="small")
        plt.savefig("cf-5-classes-" + str(n_fold) + ".png")
        plt.clf()
        plt.close()

        accuracies_test.append(accuracy_score(y[test_index], outputs_test))
        precisions.append(precision_score(y[test_index], outputs_test, average='macro', zero_division=0))
        recalls.append(recall_score(y[test_index], outputs_test, average='macro', zero_division=0))
        f1_scores.append(f1_score(y[test_index], outputs_test, average='macro', zero_division=0))

        n_fold += 1
    
    with open("results-5-classes.csv", 'w', encoding='UTF8', newline='') as csvfile:
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