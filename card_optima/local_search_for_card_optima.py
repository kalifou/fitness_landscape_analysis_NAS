from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import pickle as pkl

from absl import app
import time


#Source: http://claresloggett.github.io/python_workshops/improved_hammingdist.html

# Return the Hamming distance between string1 and string2.
# string1 and string2 should be the same length.
def hamming_distance(string1, string2): 
    # Start with a distance of zero, and count up
    distance = 0
    # Loop over the indices of the string
    L = len(string1)
    for i in range(L):
        # Add 1 to the distance if these two characters are not equal
        if string1[i] != string2[i]:
            distance += 1
    # Return the final count of differences
    return distance

def is_local_optima(all_models, regime='36', metric='kappaCohen_test'):
    
    n_models = len(all_models)
    
    is_local_optima = dict()
    Perf_nei_of_i = dict()
    
    for m_i in range(n_models):
        # all_models[m_j]['fitness'][regime][metric] if hamming_distance(m_i, m_j) == 1
        Perf_nei_of_i[m_i] = []
        for m_j in range(n_models):
            if  m_j != m_i:
                Perf_nei_of_i[m_i].append(hamming_distance(all_models[m_i]['encoded'],all_models[m_j]['encoded'] ) )
        print(sorted(Perf_nei_of_i[m_i])[0], m_i)

def collect_nei(str_current, len_encoding):
    nei_current = list()
    for i in range(len_encoding):
        nei_i = str_current.copy()
        if str_current[i] == 0:
            nei_i[i] = 1
        else: 
            nei_i[i] = 0
        nei_current.append(nei_i)
    return nei_current

def get_solution_by_value(reference, keys, dataset):
    result = dataset.copy()    
    for i in range(len(keys)):
        result = result.loc[result[keys[i]] == reference[keys[i]]]
    return result

def get_value_by_distance(reference, keys, dataset):
    result = dataset.copy()
    
    result = result.eq(reference[keys])[keys].all(1) #get_solution_by_value(item_cmp, indices, nasbench_res)
    result = result[result]
    return result

def get_solution_by_value_multiple(references, keys, dataset):
    
    list_refs = references.copy()
    len_res = len(list_refs)
    list_indices = list()
    list_results = list()
    
    for _ in range(len_res):
        list_results.append(dataset.copy())
    
    for i in range(len(keys)):
        for idx_res in range(len_res):
            result_i = list_results[idx_res]
            result_i = result_i.loc[result_i[keys[i]] == list_refs[idx_res][keys[i]]]
            list_results[idx_res] = result_i
    return list_results


def best_improvement_local_search(dataset_local, stating_point_index=88, N_max_iteration=100, muted=False):
    
    # Init starting point and counters
    current_m_idx = stating_point_index
    previous_m_idx = -1000 

    cpt = 0
    N_iter = N_max_iteration

    # Iterate for a given Budget of candidates
    while cpt < N_iter:

        # Storing previous candidate
        if previous_m_idx != current_m_idx:
            current_m_idx

        # Collect data of the current candidate    
        current_data =  dataset_local.loc[current_m_idx]
        current_perfs_36e = current_data['test_acc.36']

        #Collect neibors of the current model
        nei = collect_nei(current_data[indices], len_encoding=len(indices))

        best_item = None
        if not muted:
            print('Current idx, ',current_data.name, ', Current_perfs: ', current_perfs_36e, 
                  ', Past idx: ', previous_m_idx, '\n')

        best_index = current_m_idx
        best_perfs_36 = current_perfs_36e


        #For all neibors
        for idx, i_nei in enumerate(nei):
            
            # identify location in dataset
            sol = get_solution_by_value(i_nei, indices, dataset_local)

            #If existing, i.e valid model, do...
            if not sol.empty:

                # if many candidate with same binary, chose first
                idx_confusion = 0
                sol_i = sol.iloc[idx_confusion]

                # Update the index to neibor of better performance (Hill Climbing / Ascent)
                nei_i_perfs_36e = sol_i['test_acc.36']

                if nei_i_perfs_36e > best_perfs_36:

                    best_index = sol_i.name
                    best_perfs_36 = nei_i_perfs_36e

                    if not muted:
                        print(idx, sol.shape, nei_i_perfs_36e, sol_i.name)

        if best_index != current_m_idx and best_perfs_36 != current_perfs_36e:
            previous_m_idx = current_m_idx
            current_m_idx = best_index
            cpt += 1 
        else:
            if not muted:
                print('Step: ', cpt, ' - No improvment !\n')
            break;
    return current_m_idx, current_perfs_36e, cpt


def best_improvement_local_search_mp(dataset_local, stating_point_index=88, 
                                     N_max_iteration=100, muted=False, n_threads=4):
    
    # Init starting point and counters
    current_m_idx = stating_point_index
    previous_m_idx = -1000 

    cpt = 0
    N_iter = N_max_iteration

    # Iterate for a given Budget of candidates
    while cpt < N_iter:

        # Storing previous candidate
        if previous_m_idx != current_m_idx:
            current_m_idx

        # Collect data of the current candidate    
        current_data =  dataset_local.loc[current_m_idx]
        current_perfs_36e = current_data['test_acc.36']

        #Collect neibors of the current model
        nei = collect_nei(current_data[indices], len_encoding=len(indices))

        best_item = None
        if not muted:
            print('Current idx, ',current_data.name, ', Current_perfs: ', current_perfs_36e, 
                  ', Past idx: ', previous_m_idx, '\n')

        best_index = current_m_idx
        best_perfs_36 = current_perfs_36e
        
        with Pool(n_threads) as p:
            solutions = p.starmap(get_solution_by_value, 
                                  [(i_nei, indices, dataset_local) for i_nei in nei])
        

        #For all neibors
        for idx, sol in enumerate(solutions):
            
            # identify location in dataset
            #sol = get_solution_by_value(i_nei, indices, dataset_local)

            #If existing, i.e valid model, do...
            if not sol.empty:

                # if many candidate with same binary, chose first
                idx_confusion = 0
                sol_i = sol.iloc[idx_confusion]

                # Update the index to neibor of better performance (Hill Climbing / Ascent)
                nei_i_perfs_36e = sol_i['test_acc.36']

                if nei_i_perfs_36e > best_perfs_36:

                    best_index = sol_i.name
                    best_perfs_36 = nei_i_perfs_36e

                    if not muted:
                        print(idx, sol.shape, nei_i_perfs_36e, sol_i.name)

        if best_index != current_m_idx and best_perfs_36 != current_perfs_36e:
            previous_m_idx = current_m_idx
            current_m_idx = best_index
            cpt += 1 
        else:
            if not muted:
                print('Step: ', cpt, ' - No improvment !\n')
            break;
    return current_m_idx, current_perfs_36e, cpt


import random

def repeat_bils(dataset_local, k_starting_points=100, N_max_iteration=100, muted=False, seed=111):
    
    random.seed(seed)
    
    list_starting_points = list()
    list_ending_points = list()
    len_dataset = dataset_local.shape[0]
    
    # Sample k distinct starting points
    while len(list_starting_points) < k_starting_points:
        point = random.randint(1, len_dataset) 
        if point not in list_starting_points:
            list_starting_points.append(point)

    print('Repeating best imprving local search - k times: ', k_starting_points, '\n')
    
    # Collect k local maxima via local search
    for s_point_i in list_starting_points:        
        local_optima_i, perfs_i, steps_i = best_improvement_local_search(dataset_local=dataset_local, 
                                      stating_point_index=s_point_i, 
                                      N_max_iteration=N_max_iteration, 
                                      muted=muted)
        list_ending_points.append((local_optima_i, perfs_i, s_point_i, steps_i))
        
        print('\nOptima, perf_test_36e, Starting_point, steps:', local_optima_i, perfs_i, s_point_i, steps_i, '\n\n')
    
    return list_ending_points
        
    
import multiprocessing as mp
from multiprocessing import Pool

import random

def repeat_bils_mp(dataset_local, k_starting_points=100, N_max_iteration=100, muted=False, seed=111, n_threads=2):
    
    random.seed(seed)
    
    list_starting_points = list()
    list_ending_points = list()
    len_dataset = dataset_local.shape[0]
    
    # Sample k distinct starting points
    while len(list_starting_points) < k_starting_points:
        point = random.randint(1, len_dataset) 
        if point not in list_starting_points:
            list_starting_points.append(point)

    print('Repeating best imprving local search - k times: ', k_starting_points, '\n')
    
    
    with Pool(4) as p:
        list_ending_points = p.starmap(best_improvement_local_search, 
                             [(dataset_local.copy(), s_point_i, N_max_iteration, muted) for s_point_i in list_starting_points])

    
           
    #    print('\nOptima, perf_test_36e, Starting_point, steps:', local_optima_i, perfs_i, s_point_i, steps_i, '\n\n')
    
    return list_ending_points


if __name__ == '__main__':
    import pyreadr
    import pandas as pd
    result = pyreadr.read_r('nasbench_full_landscape_small_fixes.RData')
    result = pd.Series(result)

    nasbench_res =  result['land.df']

    len_encoding = int(17*16/2)
    len_full_solution = 155
    list_stats = list()
    indices = ['X' + str(i) for i in range(1, len_encoding + 1 )]
    
    time_begin = time.time()
    s_points=200
    n_threads_local = 4
    list_stats = repeat_bils_mp(nasbench_res, k_starting_points=s_points, muted=True, n_threads=n_threads_local)
    
    time_end = time.time()
    duration = time_end- time_begin
    print(duration, list_stats)
