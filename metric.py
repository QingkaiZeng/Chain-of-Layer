from tqdm import tqdm
from utils import *
import numpy as np
import os
import json
import random

def cal_hit_at_n(edge, filter_scores, n):
    # edge: [parent_term, child_term]
    # filter_scores: {child_term: {parent_term: score, ...}, ...}
    # n: top n
    # return: 1 or 0
    child_term = edge[1]
    parent_term = edge[0]
    if child_term not in filter_scores:
        print(filter_scores.keys())
        raise ValueError(f'child term {child_term} is not in filter scores')
    else:
        if parent_term not in filter_scores[child_term]:
            raise ValueError(f'parent term {parent_term} is not in filter scores')
        else:
            # sort filter_scores[child_term] by value
            sorted_filter_scores = sorted(filter_scores[child_term].items(), key=lambda item: item[1], reverse=True)
            #print(sorted_filter_scores)
            top_n = [item[0] for item in sorted_filter_scores[:n]]
            #print(top_n)
            if parent_term in top_n:
                return 1
            else:
                return 0

def filter_edges(edges, ground_truth_entity, filter_mode, filter_topk, filter_scores):
    if filter_mode == 'lm_score_ensemble':
        if filter_scores is None:
            raise ValueError('filter_scores is None')
        if filter_topk is None:
            raise ValueError('filter_topk is None')
    
    revised_edges = []
    for edge in edges:
        if edge[0].lower() not in ground_truth_entity or edge[1].lower()  not in ground_truth_entity:
            continue
        else:
            revised_edges.append(edge)
    
    filter_scores = {child_term.lower(): {parent_term.lower(): filter_scores[child_term][parent_term] for parent_term in filter_scores[child_term]} for child_term in filter_scores}
    if filter_mode == 'lm_score_ensemble':
        filtered_edges = []
        for edge in revised_edges:
            cal_hit_at_n_result = cal_hit_at_n(edge, filter_scores, filter_topk)
            if cal_hit_at_n_result == 1:
                filtered_edges.append(edge)
            else:
                continue
    else:
        raise NotImplementedError
            
    return filtered_edges
        

def cul_results(model_response, ground_truth_list, mode = 'edge', new_prompt = False, ChainofLayers = False, filter_mode = None, filter_topk = None, filter_scores_list = None):
    '''
    culculate the precision, recall and f1 score for the given model response and ground truth
    model_response: the response of the model
    ground_truth_list: the ground truth
    mode: 'edge' or 'entity'
        - edge: culculate the precision, recall and f1 score for the edges coverage
        - entity: culculate the precision, recall and f1 score for the entities coverage
    '''
    results = []
    results_tpfpnp = []
    for i,multi_times_response in enumerate(model_response):
        multi_times_results = []
        multi_times_results_tpfpnp = []
        gt_edges = ground_truth_list[i]
        ground_truth_entity = set()
        for par, chi in gt_edges:
            ground_truth_entity.add(par.lower())
            ground_truth_entity.add(chi.lower())
            
        gt_edges = [(each[0].lower(), each[1].lower()) for each in gt_edges]
            
        for each_time_response in multi_times_response:
            if new_prompt:
                each_edges = [each.strip().strip(';').split(' is a subtopic of ') for each in each_time_response.split(';') if ' is a subtopic of ' in each]
                each_edges = [each for each in each_edges if len(each) == 2]
                each_edges = [(par.lower(), chi.lower()) for chi, par in each_edges]
                each_entities_set = set()
                for each in each_edges:
                    each_entities_set.add(each[0])
                    each_entities_set.add(each[1])
            else:
                try:
                    each_edges, each_entities_set = phrase_taxo(each_time_response)
                
                except:
                    raise ValueError('phrase_taxo error')
            
            if len(each_entities_set) == 0:
                multi_times_results.append([0,0,0])
            else:
                if not new_prompt:
                # lower case
                    each_edges = [(each[0].lower(), each[1].lower()) for each in each_edges]
                    if not ChainofLayers and filter_mode is not None and filter_scores_list is not None:
                        each_edges = filter_edges(each_edges, ground_truth_entity, filter_mode, filter_topk, filter_scores_list[i])
                    
                    new_each_entities_set = set()
                    for edge in each_edges:
                        new_each_entities_set.add(edge[0].lower())
                        new_each_entities_set.add(edge[1].lower())
                    each_entities_set = new_each_entities_set
                
                if mode == 'edge':
                    precision, recall, f1, [TP, FP, FN] = culculate_edge_coverage(each_edges, gt_edges)
                elif mode == 'entity':
                    precision, recall, f1, [TP, FP, FN] = culculate_entity_coverage(each_entities_set, ground_truth_entity)
                elif mode == 'ancestor':
                    precision, recall, f1, [TP, FP, FN] = culculate_ancestor_coverage(each_entities_set, ground_truth_entity, each_edges, gt_edges)
                multi_times_results.append([precision, recall, f1])
                multi_times_results_tpfpnp.append([TP, FP, FN])
        results.append(multi_times_results)
        results_tpfpnp.append(multi_times_results_tpfpnp)

    avg_results = []
    for i, each_result in enumerate(results):
        avg_results.append([round(each, 4) for each in np.mean(each_result, axis = 0).tolist()])
        
    total_TP = total_FP = total_FN =0
    
    for i, each_results in enumerate(results_tpfpnp):
        #print(each_results)
        for i, each_time_results in enumerate(each_results):
            TP, FP, FN = each_time_results
            total_TP += TP
            total_FP += FP
            total_FN += FN
    try:
        total_precision = total_TP / (total_TP + total_FP)
        total_recall = total_TP / (total_TP + total_FN)
    except:
        total_precision = 0
        total_recall = 0
    try:
        total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
    except:
        total_f1 = 0

    avg_results_merge = [round(total_precision, 4), round(total_recall, 4), round(total_f1, 4)]
    
    try:
        avg_results_total = [round(each, 4) for each in np.mean(avg_results, axis = 0).tolist()]
    except:
        print(multi_times_results)
        raise ValueError('avg_results_total error')
    return avg_results_total, avg_results, results, avg_results_merge

def process_messages(messages):
    last_assistant_idx = None
    if messages[-1]['role'] != 'assistant':
        last_assistant_idx = -2
        #print(messages[-1]['content'])
        #raise ValueError('The last message should be assistant')
        if messages[-2]['role'] != 'user':
            taxo_text = messages[-2]['content']
        else:
            raise ValueError('The last message should be assistant')
    else:
        if 'the taxonomy is complete' in messages[-1]['content'].lower() and 'check: is the remaining entity list empty?' in messages[-1]['content'].lower():
            last_assistant_idx = -1
            taxo_text = messages[-1]['content'].split('Check: Is the remaining entity list empty?')[-2]
        else:
            last_assistant_idx = -3
            taxo_text = messages[-3]['content']

    """
    if 'the taxonomy is complete' not in messages[-1]['content'].lower():
        raise ValueError('The taxonomy is not complete')
    
    if messages[-2]['role'] != 'user':
        raise ValueError('The second last message should be user')
    
    if messages[-3]['role'] != 'assistant':
        raise ValueError('The third last message should be assistant')
    """
    
    
    each_edges, each_entities_set = phrase_taxo(taxo_text)
    if len(each_entities_set) == 0:
        while True:
            if last_assistant_idx-2 < -len(messages):
                break
            taxo_text = messages[last_assistant_idx-2]['content']
            each_edges, each_entities_set = phrase_taxo(taxo_text)
            if len(each_entities_set) != 0:
                break
        #raise ValueError('The taxonomy is empty')
    
    return taxo_text

def eval(taxo_name, taxo_path, model, save_path_model_response, numofExamples = 0, file_name = 'test.json', save = False, new_prompt = False, ChainofLayers = False, iteratively = False, filter_mode = None, filter_topk = None, filter_scores_list = None):
    '''
    eval the model response
    taxo_name: the name of the taxonomy
    taxo_path: the path of the taxonomy
    model: the model name
    save_path_model_response: the path to save the generated taxonomy
    numofExamples: the number of incontext examples
    file_name: the name of the file that contains the subgraphs
    save: whether to save the results
    
    printout the results
    '''
    if filter_topk is not None and ChainofLayers:
        path = save_path_model_response + taxo_name + '_top' + str(filter_topk) + '/' + model + '/' + str(numofExamples) + 'shots/'
    else:
        path = save_path_model_response + taxo_name + '/' + model + '/' + str(numofExamples) + 'shots/'
    ground_truth_list = json.load(open(path + 'ground_truth_list.json', 'r'))
    ground_truth_edge_list = []
    for each_gt in ground_truth_list:
        ground_truth_edge_list.append([tuple(each) for each in each_gt])
    if ChainofLayers and iteratively:
        model_response = json.load(open(path + 'model_response.json', 'r'))
    else:
        model_response = np.load(path + 'model_response.npy', allow_pickle=True).tolist()
    
    if ChainofLayers:
        if not iteratively:
            for i in range(len(model_response)):
                for j in range(len(model_response[i])):
                    try:
                        model_response[i][j] = remove_empty_lines(model_response[i][j].split('Check: Is the remaining entity list empty?')[-2])
                    except:
                        model_response[i][j] = model_response[i][j]
        else:
            for i in range(len(model_response)):
                for j in range(len(model_response[i])):
                    model_response[i][j] = process_messages(model_response[i][j])
    
    avg_results_total, avg_results, results, avg_results_merge = cul_results(model_response, ground_truth_edge_list, mode = 'edge', new_prompt = new_prompt, ChainofLayers = ChainofLayers, filter_mode = filter_mode, filter_topk = filter_topk, filter_scores_list = filter_scores_list)
    print('TAXO_NAME:', taxo_name, 'PATH:', save_path_model_response, 'MODEL:', model, 'NUMOFEXAMPLES:' , numofExamples)
    print('Edge Coverage:')
    if 'wordnet' in taxo_name:
        print('[p, r, f]: ', avg_results_total)
    elif 'semeval' in taxo_name:
        print('[p, r, f]: ', avg_results_total)
        print('[p, r, f]: ', avg_results)
    elif taxo_name in ['wiki_root', 'cvd_root', 'dblp_root']:
        print('[p, r, f]: ', avg_results_total)
    elif taxo_name in ['wiki', 'cvd', 'dblp']:
        print('[p, r, f]: ', avg_results_total)
        print('[p, r, f]: ', avg_results_merge)
    elif taxo_name in ['wiki_downsample', 'cvd_downsample', 'dblp_downsample', 'dblp_sampled_downsample', 'cvd_sampled_downsample']:
        print('[p, r, f]: ', avg_results_total)
        print('[p, r, f]: ', avg_results)
    elif taxo_name in ['cvd_single']:
        print('[p, r, f]: ', avg_results_total)
        #print(avg_results)
        
    
    avg_results_total, avg_results, results, avg_results_merge = cul_results(model_response, ground_truth_list, mode = 'ancestor', new_prompt = new_prompt, ChainofLayers = ChainofLayers, filter_mode = filter_mode, filter_topk = filter_topk, filter_scores_list = filter_scores_list)
    print('Ancestor Coverage:')
    if 'wordnet' in taxo_name:
        print('[p, r, f]: ', avg_results_total)
    elif 'semeval' in taxo_name:
        print('[p, r, f]: ', avg_results_total)
        print('[p, r, f]: ', avg_results)
    elif taxo_name in ['wiki_root', 'cvd_root', 'dblp_root']:
        print('[p, r, f]: ', avg_results_total)
    elif taxo_name in ['wiki', 'cvd', 'dblp']:
        print('[p, r, f]: ', avg_results_merge)
    elif taxo_name in ['wiki_downsample', 'cvd_downsample', 'dblp_downsample', 'dblp_sampled_downsample', 'cvd_sampled_downsample']:
        print('[p, r, f]: ', avg_results_total)
    elif taxo_name in ['cvd_single']:
        print('[p, r, f]: ', avg_results_total)
    
    avg_results_total, avg_results, results, avg_results_merge = cul_results(model_response, ground_truth_list, mode = 'entity', new_prompt = new_prompt, ChainofLayers = ChainofLayers, filter_mode = filter_mode, filter_topk = filter_topk, filter_scores_list = filter_scores_list)
    print('Entity Coverage:')
    if 'wordnet' in taxo_name:
        print('[p, r, f]: ', avg_results_total)
    elif 'semeval' in taxo_name:
        print('[p, r, f]: ', avg_results_total)
        print('[p, r, f]: ', avg_results)
    elif taxo_name in ['wiki_root', 'cvd_root', 'dblp_root']:
        print('[p, r, f]: ', avg_results_total)
    elif taxo_name in ['wiki', 'cvd', 'dblp']:
        print('[p, r, f]: ', avg_results_merge)
    elif taxo_name in ['wiki_downsample', 'cvd_downsample', 'dblp_downsample', 'dblp_sampled_downsample', 'cvd_sampled_downsample']:
        print('[p, r, f]: ', avg_results_total)
    elif taxo_name in ['cvd_single']:
        print('[p, r, f]: ', avg_results_total)    

    if save:
        raise NotImplementedError
    
    print()
    
def eval_analysis_num_entity(taxo_name, taxo_path, model, save_path_model_response, numofExamples = 0, file_name = 'test.json', save = False, new_prompt = False, ChainofLayers = False, iteratively = False, filter_mode = None, filter_topk = None, filter_scores_list = None):
    assert filter_topk is not None
    for i in ['20', '40', '60', '80', '100', '120', '140', '160']:
        path = save_path_model_response + taxo_name + '_' + i + f'_top{filter_topk}/' + model + '/' + str(numofExamples) + 'shots/'
        ground_truth_list = json.load(open(path + 'ground_truth_list.json', 'r'))
        ground_truth_edge_list = []
        for each_gt in ground_truth_list:
            ground_truth_edge_list.append([tuple(each) for each in each_gt])
        if ChainofLayers and iteratively:
            model_response = json.load(open(path + 'model_response.json', 'r'))
        else:
            model_response = np.load(path + 'model_response.npy', allow_pickle=True).tolist()
        
        if ChainofLayers:
            if not iteratively:
                for i in range(len(model_response)):
                    for j in range(len(model_response[i])):
                        try:
                            model_response[i][j] = remove_empty_lines(model_response[i][j].split('Check: Is the remaining entity list empty?')[-2])
                        except:
                            model_response[i][j] = model_response[i][j]
            else:
                for i in range(len(model_response)):
                    for j in range(len(model_response[i])):
                        model_response[i][j] = process_messages(model_response[i][j])
        
        avg_results_total, avg_results, results, avg_results_merge = cul_results(model_response, ground_truth_edge_list, mode = 'edge', new_prompt = new_prompt)
        print('TAXO_NAME:', taxo_name, 'PATH:', save_path_model_response, 'MODEL:', model, 'NUMOFEXAMPLES:' , numofExamples)
        print('Edge Coverage:')
        print('[p, r, f]: ', avg_results_total)

            
        avg_results_total, avg_results, results, avg_results_merge = cul_results(model_response, ground_truth_list, mode = 'ancestor', new_prompt = new_prompt)
        print('Ancestor Coverage:')
        print('[p, r, f]: ', avg_results_total)
        
        avg_results_total, avg_results, results, avg_results_merge = cul_results(model_response, ground_truth_list, mode = 'entity', new_prompt = new_prompt)
        print('Entity Coverage:')
        print('[p, r, f]: ', avg_results_total)


        if save:
            raise NotImplementedError
    