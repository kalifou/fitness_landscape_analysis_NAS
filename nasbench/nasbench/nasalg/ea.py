from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random

from deap import base
from deap import creator
from deap import tools

import numpy as np

from absl import app
from nasbench import api


INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OPERATIONS = [INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT]

MAX_EDGES = 9
MAX_NODES = 7

NASBENCH_TFRECORD = './data/nasbench_full.tfrecord'
NASBENCH = api.NASBench(NASBENCH_TFRECORD)


def _rec_valid_paths(matrix, path=[]):
    if path[-1][1] == (matrix.shape[1] - 1):
        return path
    # where are we going next?
    rec = np.nonzero(matrix[path[-1][1], :])[0]
    if len(rec) == 0:
        # Dead end
        return []
    else:
        output = list()
        for dy in rec:
            p = _rec_valid_paths(matrix, path + [(path[-1][1], dy)])
            if len(p) > 0:
                if isinstance(p[0], list):
                    output = output + p
                else:
                    output = output + [p]
        return output


def valid_paths(matrix):
    paths = list() 
    for iy in np.nonzero(matrix[0, :])[0]:
        p = _rec_valid_paths(matrix, [(0, iy)])
        if len(p) > 0:
            if isinstance(p[0], list):
                paths = paths + p
            else:
                paths = paths + [p]
    return paths


def add_valid_path(matrix):
    # We don't want to connect the input directly to the output, i.e., MAX_NODES - 1
    p0 = np.random.choice(np.arange(1, MAX_NODES - 1))
    matrix[0, p0] = 1
    px = p0
    while True:
        if _rec_valid_paths(matrix, [(0, p0)]):
            break
        py = np.random.choice(np.arange(px + 1, MAX_NODES))
        matrix[px, py] = 1
        px = py
    return matrix


def check_and_repair(matrix):
    # Get the valid paths
    paths = valid_paths(matrix)
    # Get the visited nodes/edges
    visited_nodes = set([item for sublist in paths for item in sublist])
    # Remove unvisited edges from the matrix
    ix, iy = np.nonzero(matrix)
    for pos in zip(ix, iy):
        if pos not in visited_nodes:
            matrix[pos] = 0
    # Check that the number of edges is within the constraints
    while np.sum(matrix) > MAX_EDGES:
        # Randomly remove one edge
        ix, iy = np.nonzero(matrix)
        del_pos = np.random.choice(len(ix))
        matrix[ix[del_pos], iy[del_pos]] = 0
        matrix = check_and_repair(matrix)
    # TODO: Check that the model is available in NASBENCH?
    if np.sum(matrix) == 0:
        matrix = add_valid_path(matrix)
    return matrix


def ind_to_matrix(ind):
    matrix = np.triu(np.ones(MAX_NODES), k=1)
    matrix[matrix > 0] = ind
    matrix[0, MAX_NODES-1] = 0 # It is not possible to connect the input with the output
    return matrix.astype(int)


def update_ind(matrix, ind):
    valid_ind = matrix[np.triu_indices(MAX_NODES, k=1)]
    for i in range(len(ind)):
        ind[i] = int(valid_ind[i])
    return ind
    

# Check that a solution is valid
def check_and_repair_individual(individual):
    matrix = ind_to_matrix(individual)
    matrix = check_and_repair(matrix)
    individual = update_ind(matrix, individual)
    return individual


def single_point_path_xo(ind1, ind2):
    # Get the visited nodes/edges from ind1
    matrix1 = ind_to_matrix(ind1)
    paths1 = valid_paths(matrix1)    
    # Repeat for ind2
    matrix2 = ind_to_matrix(ind2)
    paths2 = valid_paths(matrix2)
    # Cross-over the two individuals
    if len(paths1) == 0 or len(paths2) == 0:
        return ind1, ind2
    p1 = np.random.choice(len(paths1))
    p2 = np.random.choice(len(paths2))
    if p1 == 0 and p2 == 0:
        p1 = 1
    # Get the paths
    up_paths1 = paths1[:p1] + paths2[p2:]
    visited_nodes1 = set([item for sublist in up_paths1 for item in sublist])
    matrix = np.zeros([MAX_NODES, MAX_NODES])
    for pos in visited_nodes1:
        matrix[pos] = 1
    ind1 = update_ind(matrix, ind1)
    # Now, for the second ind
    up_paths2 = paths1[p1:] + paths2[:p2]
    visited_nodes2 = set([item for sublist in up_paths2 for item in sublist])
    matrix = np.zeros([MAX_NODES, MAX_NODES])
    for pos in visited_nodes2:
        matrix[pos] = 1
    ind2 = update_ind(matrix, ind2)
    return ind1, ind2


def mutate_path(ind):
    matrix = ind_to_matrix(ind)
    if np.random.rand() > 0.5:
        # Remove one edge. Note that during the check, at least one path will be removed
        ix, iy = np.nonzero(matrix)
        if len(ix) > 0:
            pdel = np.random.choice(len(ix))
            matrix[ix[pdel], iy[pdel]] = 0
    else:
        # Add a valid path
        matrix = add_valid_path(matrix)
    ind = update_ind(matrix, ind)
    return ind


# the goal ('fitness') function to be maximized
def eval_on_benchmark(individual):
    matrix = ind_to_matrix(individual)
    model_spec = api.ModelSpec(
        matrix,
        ops=OPERATIONS)
    try:
        data = NASBENCH.query(model_spec)
    except:
        print("Model not found\n", matrix)
        data = {'test_accuracy': 0}
    return data['test_accuracy'],


# e.g., 'trainable_parameters': 2694282, 'training_time': 1154.361083984375, 'train_accuracy': 1.0, 'validation_accuracy': 0.9336938858032227, 'test_accuracy': 0.9286859035491943
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)

# Adjacency matrix of the module
#      matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
#              [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
#              [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
#              [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
#              [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
#              [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
#              [0, 0, 0, 0, 0, 0, 0]],   # output layer
# Operations at the vertices of the module, matches order of matrix
#      ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])
# upper-diagonal matrix, i.e., individual=<g_1, ..., g_21>, g_i={0,1}, sum(individual)<=9 
#      matrix=[[, 0, 1,  2,  3,  4,  5],    # input layer
#              [,  , 6,  7,  8,  9, 10],    # 1x1 conv
#              [,  ,  , 11, 12, 13, 14],    # 3x3 conv
#              [,  ,  ,   , 15, 16, 17],    # 5x5 conv (replaced by two 3x3's)
#              [,  ,  ,   ,   , 18, 19],    # 5x5 conv (replaced by two 3x3's)
#              [,  ,  ,   ,   ,   , 20],    # 3x3 max-pool
#              [,  ,  ,   ,   ,   ,  ]],   # output layer
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 21)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_on_benchmark)

# single point path crossover
toolbox.register("mate", single_point_path_xo)

# bit fplip mutation wtih probability indpb
toolbox.register("mutate", mutate_path)

# binary tournament selection
toolbox.register("select", tools.selTournament, tournsize=2)

# check and repair individuals
toolbox.register("check", check_and_repair_individual)


def ga_run(hof_size=1, seed=20172019):
    random.seed(seed)
    np.random.seed(seed)

    hall_of_fame = tools.HallOfFame(hof_size)

    # create an initial population of n individuals
    pop = toolbox.population(n=10)
    pop = list(map(toolbox.check, pop))

    # Cross-over probability
    CXPB = 0.5
    # Mutation probability
    MUTPB = 2 / (MAX_NODES * (MAX_NODES - 1))
    MAX_EVALS = 100

    print("--  GA (random seed= %d)  --" % seed)
    evals = 0
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hall_of_fame.update(pop)
    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    evals = len(pop)    
    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5

    g = 0
    print("**  Gen %d  Evals %d  Min %s  Max %s  Avg %s  Std %s" % (g, evals, min(fits), max(fits), mean, std))

    while max(fits) < 1 and evals < MAX_EVALS:
        # A new generation
        g = g + 1
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        pop = list(map(toolbox.check, offspring))
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            if evals < MAX_EVALS:
                ind.fitness.values = toolbox.evaluate(ind)
                evals = evals + 1
               
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        hall_of_fame.update(pop)
        
        # Gather all the fitnesses in one list and print the stats
        
        fits = [ind.fitness.values[0] if ind.fitness.valid else np.nan for ind in pop]
        print("**  Gen %d  Evals %d  Min %s  Max %s  Avg %s  Std %s" %
                (g, evals, np.nanmin(fits), np.nanmax(fits), np.nanmean(fits), np.nanstd(fits)))

    
    print("--  End of evolution (random seed= %d)  --" % seed)
    
    # best_ind = tools.selBest(pop, 1)[0]
    # print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    for ind in hall_of_fame:
        print("HOF  %s, %s" % (ind, ind.fitness.values))


def random_search_run(hof_size=1, seed=20172019):
    random.seed(seed)
    np.random.seed(seed)

    hall_of_fame = tools.HallOfFame(hof_size)

    # create an initial population of n individuals
    pop = toolbox.population(n=100)
    pop = list(map(toolbox.check, pop))

    print("--  RS (random seed= %d)  --" % seed)
    evals = 0
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hall_of_fame.update(pop)
    evals = len(pop) 
    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] if ind.fitness.valid else np.nan for ind in pop]

    print("**  Evals %d  Min %s  Max %s  Avg %s  Std %s" %
            (evals, np.nanmin(fits), np.nanmax(fits), np.nanmean(fits), np.nanstd(fits)))
   
    print("--  End of RS (random seed= %d)  --" % seed)
    
    # best_ind = tools.selBest(pop, 1)[0]
    # print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    for ind in hall_of_fame:
        print("HOF  %s, %s" % (ind, ind.fitness.values))


if __name__ == "__main__":
    init_seed = 20172019
    runs = 30
    for seed in range(init_seed, (init_seed + runs)) :
        #ga_run(seed=seed)
        random_search_run(seed=seed)
