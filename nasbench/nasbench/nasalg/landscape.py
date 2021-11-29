from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json

from absl import app
from nasbench import api

MAX_NODES = 7
MAX_EDGES = 9
NASBENCH_TFRECORD = '/local_home/trao_ka/projects/nasbench/nasbench_full.tfrecord'

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

CODING = [INPUT]
CODING = CODING + [CONV1X1 + "_" + str(i) for i in range(0, (MAX_NODES - 2))]
CODING = CODING + [CONV3X3 + "_" + str(i) for i in range(0, (MAX_NODES - 2))]
CODING = CODING + [MAXPOOL3X3 + "_" + str(i) for i in range(0, (MAX_NODES - 2))]
CODING = CODING + [OUTPUT]


def rename_ops(ops):
    c1x1 = 0
    c3x3 = 0
    mp3x3 = 0
    new_ops = []
    for op in ops:
        if op == CONV1X1:
            new_ops = new_ops + [op + "_" + str(c1x1)]
            c1x1 = c1x1 + 1
        elif op == CONV3X3:
            new_ops = new_ops + [op + "_" + str(c3x3)]
            c3x3 = c3x3 + 1
        elif op == MAXPOOL3X3:
            new_ops = new_ops + [op + "_" + str(mp3x3)]
            mp3x3 = mp3x3 + 1
        else:
            new_ops = new_ops + [op]
    return new_ops


def encode_matrix(adj_matrix, ops):
    enc_matrix = np.zeros((len(CODING), len(CODING)))
    pos = [CODING.index(op) for op in ops]
    trans = dict()
    for i, ix in enumerate(pos):
        trans[i] = ix
    i, j = np.nonzero(adj_matrix)
    ix = [trans.get(n) for n in i]
    jy = [trans.get(n) for n in j]
    for p in zip(ix, jy):
        enc_matrix[p] = 1
    encoded = enc_matrix[np.triu_indices(len(CODING), k=1)]
    return encoded.astype(int)


def encode_solution(solution):
    adj_matrix = solution['module_adjacency']
    ops = rename_ops(solution['module_operations'])
    encoded = encode_matrix(adj_matrix, ops)    
    return encoded, solution['trainable_parameters'], ops


def summarize_fitness(computed_metrics, epochs=[108]):
    fitness = dict()
    for ep in epochs:
        training_time = 0
        train_acc = 0
        validation_acc = 0
        test_acc = 0
        for metrics in computed_metrics[ep]:
            training_time = metrics['final_training_time']
            train_acc = train_acc + metrics['final_train_accuracy']
            validation_acc = validation_acc + metrics['final_validation_accuracy']
            test_acc = test_acc + metrics['final_test_accuracy']
        training_time = training_time / len(computed_metrics[ep])
        train_acc = train_acc / len(computed_metrics[ep])
        validation_acc = validation_acc / len(computed_metrics[ep])
        test_acc = test_acc / len(computed_metrics[ep])
        fitness[ep] = {
            'training_time': training_time,
            'train_acc': train_acc,
            'validation_acc': validation_acc,
            'test_acc': test_acc}
    return fitness




def get_fitnesses():
    nasbench = api.NASBench(NASBENCH_TFRECORD)
    prev = ""
    print('{\n\t"coding": ', json.dumps(CODING), ",")
    print('\t"solutions": [\n')
    for unique_hash in nasbench.hash_iterator():
        fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)
        encoded, params, ops = encode_solution(fixed_metrics)
        fitness = summarize_fitness(computed_metrics, epochs=[4, 12, 36, 108])
        print('\t\t', prev, '{"encoded": ', encoded.tolist(),
              ', "tr_params": ', params, 
              ', "ops": ', json.dumps(ops), 
              ', "fitness": ', json.dumps(fitness), '}')
        if not prev:
            prev = ","
    print('\t]\n}')


if __name__ == "__main__":
    get_fitnesses()
