from matplotlib import pyplot as plt
import neat
from neat import config
import numpy as np
import scikitplot
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
import os
import multiprocessing
import csv   
    
if __name__=='__main__':
    x_datasets = ['CFS-x.pickle', 'FCBF-x.pickle', 
            'MRMR-x.pickle', 'ReliefF-x.pickle']

    y_datasets = ['CFS-y.pickle', 'FCBF-y.pickle', 
            'MRMR-y.pickle', 'ReliefF-y.pickle']

    for x_dataset, y_dataset in zip(x_datasets, y_datasets):

        # Creating dataset folder

        dirname = x_dataset.split('-')
        dirname = dirname[0]

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # Import datasets

        x_data = pickle.load(open("../datasets/" + x_dataset, 'rb'))
        y_data = pickle.load(open("../datasets/" + y_dataset, 'rb'))

        y = []

        for o in y_data:
            if o == 0:
                y.append([1, 0, 0, 0, 0, 0, 0])
            if o == 1:
                y.append([0, 1, 0, 0, 0, 0, 0])
            if o == 2:
                y.append([0, 0, 1, 0, 0, 0, 0])
            if o == 3:
                y.append([0, 0, 0, 1, 0, 0, 0])
            if o == 4:
                y.append([0, 0, 0, 0, 1, 0, 0])
            if o == 5:
                y.append([0, 0, 0, 0, 0, 1, 0])
            if o == 6:
                y.append([0, 0, 0, 0, 0, 0, 1])

        x_data = np.array(x_data)
        y_data = np.array(y_data)
        y = np.array(y)

        # Modifying neat config file

        file = []

        with open('config-feedforward.txt', 'r') as f:
            file = f.readlines()
        
        file[49] = 'num_inputs              = ' + str(len(x_data[0])) + '\n'

        with open('config-feedforward.txt', 'w') as f:
            f.writelines(file)

        # Cross-validation

        accuracies_train = []
        accuracies_test = []
        balanced_accuracies = []
        precisions = []
        recalls =[]
        f1_scores = []

        n_fold = 1
        kf = StratifiedKFold(n_splits=10)
        print("Classifying ", dirname, " dataset")

        for train_index, test_index in kf.split(x_data, y_data):
            
            def eval_genomes(genome, config):
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                outputs = []

                for xi in x_data[train_index]:
                    outputs.append(np.argmax(net.activate(xi)))
                
                return precision_score(y_data[train_index], outputs, average='micro', zero_division=0)

            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                'config-feedforward.txt')
            p = neat.Population(config)
            p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)

            pe = neat.ParallelEvaluator(12, eval_genomes)
            winner = p.run(pe.evaluate, 1000)
            
            winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

            outputs_train = []
            for xi in x_data[train_index]:
                outputs_train.append(np.argmax(winner_net.activate(xi)))

            accuracies_train.append(accuracy_score(y_data[train_index], outputs_train))
            
            outputs_test = []
            for xi in x_data[test_index]:
                outputs_test.append(np.argmax(winner_net.activate(xi)))

            scikitplot.metrics.plot_confusion_matrix(y_data[test_index], outputs_test, title="Fold " + str(n_fold) + " Confusion Matrix" , normalize=True, text_fontsize="small")
            plt.savefig(os.path.join(dirname, dirname + "-confusion-matrix-" + str(n_fold) + ".png"))
            plt.clf()
            plt.close()

            with open(os.path.join(dirname, dirname + '-winner-' + str(n_fold)), 'wb') as f:
                pickle.dump(winner_net, f)
            
            accuracies_test.append(accuracy_score(y_data[test_index], outputs_test))
            balanced_accuracies.append(balanced_accuracy_score(y_data[test_index], outputs_test))
            precisions.append(precision_score(y_data[test_index], outputs_test, average='macro', zero_division=0))
            recalls.append(recall_score(y_data[test_index], outputs_test, average='macro', zero_division=0))
            f1_scores.append(f1_score(y_data[test_index], outputs_test, average='macro', zero_division=0))

            n_fold += 1

        with open(dirname + "results.csv", 'w', encoding='UTF-8', newline='') as cvsfile:
            writer = csv.writer(cvsfile)
            fold = 1
            header = ['fold', 'accuracy train', 'accuracy test', 'balanced accuracy', 'precision', 'recall', 'f1_score']
            writer.writerow(header)
            for atrain, atest, ba, p, r, f1 in zip(accuracies_train,
                                        accuracies_test, 
                                        balanced_accuracies,
                                        precisions,
                                        recalls,
                                        f1_scores):
                row = [fold, atrain, atest, ba, p, r, f1]
                writer.writerow(row)
                fold += 1

            avgs = ["Average",
                    np.mean(np.array(accuracies_train)),
                    np.mean(np.array(accuracies_test)),
                    np.mean(np.array(balanced_accuracies)),
                    np.mean(np.array(precisions)),
                    np.mean(np.array(recalls)),
                    np.mean(np.array(f1_scores)) ]
            writer.writerow(avgs)