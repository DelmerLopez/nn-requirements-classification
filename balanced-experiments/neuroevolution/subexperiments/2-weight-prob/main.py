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

    x_datasets = ['x-no-1', 'x-no-4', 'x-no-2', 'x-no-0']
    y_datasets = ['y-no-1', 'y-no-4', 'y-no-2', 'y-no-0']
    weights = [30, 20, 20, 10]
    num_classes = [6, 5, 4, 3]

    for x_dataset, y_dataset, weight, num_class in zip(x_datasets, y_datasets, weights, num_classes):

        x = pickle.load(open('../../../datasets/' + x_dataset, 'rb'))
        y = pickle.load(open('../../../datasets/' + y_dataset, 'rb'))

        mutation_p = [0.2, 0.4, 0.6]

        for m_number in mutation_p:

            # Modifying neat config file

            file = []

            with open('config-feedforward-neuroevolution.txt', 'r') as f:
                file = f.readlines()

            file[64] = 'weight_max_value        = ' + str(weight) + '\n'
            file[65] = 'weight_min_value        = ' + str(weight * -1) + '\n'
            file[67] = 'weight_mutate_rate      = ' + str(m_number) + '\n'

            with open('config-feedforward-neuroevolution.txt', 'w') as f:
                f.writelines(file)


            dirname = os.path.join(str(num_class) + "-classes", str(m_number) + "-wp")

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

                scikitplot.metrics.plot_confusion_matrix(y[test_index], outputs_test, title=str(num_class) + " Classes Confusion Matrix: Fold " + str(n_fold) , normalize=True, text_fontsize="small")
                plt.savefig(os.path.join(dirname, str(m_number)+"-wp-cf-"+ str(num_class) + "-" + str(n_fold) + ".png"))
                plt.clf()
                plt.close()

                with open(os.path.join(dirname, str(m_number) + '-winner-' + str(n_fold)), 'wb') as f:
                    pickle.dump(winner, f)
                
                with open(os.path.join(dirname, str(m_number) + 'results-' + str(n_fold)), 'w') as f:
                    f.write('\nBest genome:\n{!s}'.format(winner))

                accuracies_test.append(accuracy_score(y[test_index], outputs_test))
                precisions.append(precision_score(y[test_index], outputs_test, average='macro', zero_division=0))
                recalls.append(recall_score(y[test_index], outputs_test, average='macro', zero_division=0))
                f1_scores.append(f1_score(y[test_index], outputs_test, average='macro', zero_division=0))
                
                n_fold += 1
            
            with open(os.path.join(dirname, str(m_number) + "-wp-results-4-classes.csv"), 'w', encoding='UTF8', newline='') as csvfile:
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