import csv
import multiprocessing
import neat
import numpy as np
import os
import pickle
import scikitplot
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score


if __name__=='__main__':

    x_datasets = ['x-no-0', 'x-no-1', 'x-no-2', 'x-no-4']
    y_datasets = ['y-no-0', 'y-no-1', 'y-no-2', 'y-no-4']

    for x_dataset, y_dataset in zip(x_datasets, y_datasets):

        x = pickle.load(open('../../../datasets/' + x_dataset, 'rb'))
        y = pickle.load(open('../../../datasets/' + y_dataset, 'rb'))

        mutation_p = [5, 10, 20, 30]

        for m_number in mutation_p:

            # Modifying neat config file

            file = []

            with open('config-feedforward-neuroevolution.txt', 'r') as f:
                file = f.readlines()
            
            file[64] = 'weight_max_value        = ' + str(m_number) + '\n'
            file[65] = 'weight_min_value        = ' + str(m_number * -1) + '\n'

            with open('config-feedforward-neuroevolution.txt', 'w') as f:
                f.writelines(file)


            dirname = os.path.join(x_dataset, str(m_number) + "-hn")

            if not os.path.exists(dirname):
                os.makedirs(dirname)

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
                    
                    return accuracy_score(y[train_index], outputs)
                
                config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                    'config-feedforward-neuroevolution.txt')
                p = neat.Population(config)
                p.add_reporter(neat.StdOutReporter(True))
                stats = neat.StatisticsReporter()
                p.add_reporter(stats)

                pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genomes)
                winner = p.run(pe.evaluate, 1500)
                
                winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

                outputs_train = []
                for xi in x[train_index]:
                    outputs_train.append(np.argmax(winner_net.activate(xi)))
                
                accuracies_train.append(accuracy_score(y[train_index], outputs_train))

                outputs_test = []
                for xi in x[test_index]:
                    outputs_test.append(np.argmax(winner_net.activate(xi)))

                scikitplot.metrics.plot_confusion_matrix(y[test_index], outputs_test, title="Four Classes Confusion Matrix: Fold " + str(n_fold) , normalize=True, text_fontsize="small")
                plt.savefig(os.path.join(dirname, str(m_number)+"-hn-cf-4-classes-" + str(n_fold) + ".png"))
                plt.clf()
                plt.close()

                with open(os.path.join(dirname, str(m_number) + '-winner-' + str(n_fold)), 'wb') as f:
                    pickle.dump(winner_net, f)

                accuracies_test.append(accuracy_score(y[test_index], outputs_test))
                precisions.append(precision_score(y[test_index], outputs_test, average='macro', zero_division=0))
                recalls.append(recall_score(y[test_index], outputs_test, average='macro', zero_division=0))
                f1_scores.append(f1_score(y[test_index], outputs_test, average='macro', zero_division=0))
                
                n_fold += 1
            
            with open(os.path.join(dirname, str(m_number) + "-hn-results-4-classes.csv"), 'w', encoding='UTF8', newline='') as csvfile:
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