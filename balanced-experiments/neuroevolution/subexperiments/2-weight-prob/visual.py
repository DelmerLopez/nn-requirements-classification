import pickle
import neat
import neat.visualize as vs
import os

def read_genome(genome_file):
    file = open(genome_file, 'r')

classes = ['3-classes', '4-classes', '5-classes', '6-classes']
experiments = [0.2, 0.4, 0.6]
weights = [30, 20, 20, 10]

for clas, weight in zip(classes, weights):
    for experiment in experiments:
        
        dirname = os.path.join(clas, str(experiment) + '-wp', 'topogies')

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        file = []
        
        with open('config-feedforward-neuroevolution.txt', 'r') as f:
            file = f.readlines()

        file[64] = 'weight_max_value        = ' + str(weight) + '\n'
        file[65] = 'weight_min_value        = ' + str(weight * -1) + '\n'
        file[67] = 'weight_mutate_rate      = ' + str(experiment) + '\n'

        with open('config-feedforward-neuroevolution.txt', 'w') as f:
            f.writelines(file)
        
        for i in range(1, 11):            
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                    'config-feedforward-neuroevolution.txt')

            model_name = str(experiment) + '-winner-' + str(i)
            model_dir = os.path.join(clas, str(experiment) + '-wp', model_name)
            model = pickle.load(open(model_dir, 'rb'))
            node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
            vs.draw_net(config, model, True, filename=str(clas) + str(experiment) + '-winner-' + str(i), node_names=node_names, prune_unused=True)