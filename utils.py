from openai import OpenAI
from tqdm import tqdm
import os
import json
import numpy as np
import time
import networkx as nx

def remove_empty_lines(string):
    return os.linesep.join([s for s in string.splitlines() if s])

def count_dot(s):
    '''
    count the number of dots in the string
    '''
    return len(s) - len(s.replace(".", ""))

def phrase_taxo(raw_taxonomy):
    '''
    phrase the taxonomy
    raw_taxonomy: the raw taxonomy
    return: the edges and entities set of the taxonomy
    '''
    raw_taxonomy = raw_taxonomy.replace('I.', '1.')
    raw_taxonomy = raw_taxonomy.replace('\u200b', '')
    raw_taxonomy = raw_taxonomy.replace('I1.', '2.')
    raw_taxonomy = raw_taxonomy.replace('I11.', '3.')
    taxonomy_lines = raw_taxonomy.split("\n")
    find_idx = 0
    for i in range(len(taxonomy_lines)):
        if taxonomy_lines[i].startswith("1"):
            find_idx = i
            break

    taxonomy_lines = taxonomy_lines[find_idx:]
    remove_start = taxonomy_lines
    
    if taxonomy_lines[0].startswith("1.") and len(taxonomy_lines[0].split('1.1 ')) == 2:
        taxonomy_lines[0] = taxonomy_lines[0].split('1.1 ')[0]
        # add 1.1
        try:
            taxonomy_lines.insert(1, '1.1 ' + taxonomy_lines[0].split('1. ')[1])
        except IndexError:
            return [], set()
    # remove empty lines
    taxonomy_lines = [line.strip() for line in taxonomy_lines if line.strip()]
    remove_empty1 = taxonomy_lines
    taxonomy_lines = [line for line in taxonomy_lines if line[0].isdigit()]

    data = [line.strip().split(" ", 1) for line in taxonomy_lines]

    #print(data)
    edges = []
    d = {}
    entities_set = set()

    for i,item in enumerate(data):
        if len(item) != 2:
            continue
        index = item[0]
        if count_dot(index) == 0:
            index = index + '.'
        elif count_dot(index) == 1:
            parts = index.split('.')
            if parts[-1] == '':
                index = index
        elif count_dot(index) >= 2 and index[-1] == '.':
            index = index[:-1]
        name = item[1]
        name = name.lower().replace('_', ' ')
        entities_set.add(name)
        d[index] = name
        parts = index.split('.')
        if i == 0:
            continue

        if len(parts) > 1:
            if len(parts) == 2:
                parent_index = parts[0]+'.'
                try:
                    parent_name = d[parent_index]
                except KeyError:
                    '''
                    print(raw_taxonomy)
                    print(remove_start)
                    print(remove_empty1)
                    print(taxonomy_lines)
                    print(data)
                    print(d)
                    '''
                    return edges, entities_set
                if parent_name == name:
                    continue
                edges.append((parent_name, name))
            else:
                parent_index = '.'.join(parts[:-1])
                try:
                    parent_name = d[parent_index]
                except KeyError:
                    '''
                    print(raw_taxonomy)
                    print(remove_start)
                    print(remove_empty1)
                    print(taxonomy_lines)
                    print(data)
                    print(d)
                    '''
                    return edges, entities_set
                if parent_name == name:
                    continue
                edges.append((parent_name, name))
    return edges, entities_set

def culculate_edge_coverage(model_output, ground_truth):
    '''
    culculate the precision, recall and f1 score for the given model response and ground truth (entity coverage)
    model_output: the response of the model
    ground_truth: the ground truth
    
    return: the precision, recall and f1 score
    '''
    model_output_entity = set(model_output)
    ground_truth_entity = set(ground_truth)
    
    TP = len(model_output_entity.intersection(ground_truth_entity))
    FP = len(model_output_entity) - TP
    FN = len(ground_truth_entity) - TP
    
    if len(model_output_entity) == 0:
        return 0, 0, 0, [TP, FP, FN]

    
    coverage_r = len(model_output_entity.intersection(ground_truth_entity)) / len(ground_truth_entity)
    coverage_p = len(model_output_entity.intersection(ground_truth_entity)) / len(model_output_entity)
    try:
        coverage_f1 = 2 * coverage_r * coverage_p / (coverage_r + coverage_p)
    except ZeroDivisionError:
        coverage_f1 = 0
    return coverage_p ,coverage_r, coverage_f1, [TP, FP, FN]

def culculate_entity_coverage(model_output, ground_truth):
    '''
    culculate the precision, recall and f1 score for the given model response and ground truth (entity coverage)
    model_output: the response of the model
    ground_truth: the ground truth
    
    return: the precision, recall and f1 score
    '''
    model_output_entity = list(set(model_output))
    ground_truth_entity = list(set(ground_truth))
    
    model_output_entity = set([each.lower().strip(' ').strip('\n') for each in model_output_entity])
    ground_truth_entity = set([each.lower().strip(' ').strip('\n') for each in ground_truth_entity])
    
    TP = len(model_output_entity.intersection(ground_truth_entity))
    FP = len(model_output_entity) - TP
    FN = len(ground_truth_entity) - TP
    
    if len(model_output_entity) == 0:
        return 0, 0, 0, [TP, FP, FN]
    
    coverage_r = len(model_output_entity.intersection(ground_truth_entity)) / len(ground_truth_entity)
    coverage_p = len(model_output_entity.intersection(ground_truth_entity)) / len(model_output_entity)
    try:
        coverage_f1 = 2 * coverage_r * coverage_p / (coverage_r + coverage_p)
    except ZeroDivisionError:
        coverage_f1 = 0
    return coverage_p ,coverage_r, coverage_f1, [TP, FP, FN]

def build_graph(nodes, edges):
    '''Build a graph given the relations in df (a pandas dataframe).'''
    G = nx.DiGraph()
    edges = set(edges)
    nodes = set(nodes)
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    return G


def convert_to_ancestor_graph(G):
    '''Converts a (parent) tree to a graph with edges for all ancestor relations in the tree.'''
    G_anc = nx.DiGraph()
    for node in G.nodes():
        for anc in nx.ancestors(G, node):
            G_anc.add_edge(anc, node)
    return G_anc

def culculate_ancestor_coverage(model_output_node, ground_truth_node, model_output_edge, ground_truth_edge):
    '''
    culculate the precision, recall and f1 score for the given model response and ground truth (ancestor coverage)
    model_output: the response of the model
    ground_truth: the ground truth
    
    return: the precision, recall and f1 score
    '''
    model_out_node = set(model_output_node)
    ground_truth_node = set(ground_truth_node)
    model_output_edge = set(model_output_edge)
    ground_truth_edge = set(ground_truth_edge)
    ground_truth_G = build_graph(ground_truth_node, ground_truth_edge)
    model_output_G = build_graph(model_output_node, model_output_edge)
    ground_truth_G_anc = convert_to_ancestor_graph(ground_truth_G)
    model_output_G_anc = convert_to_ancestor_graph(model_output_G)
    model_output_G_anc_edge = set(model_output_G_anc.edges())
    ground_truth_G_anc_edge = set(ground_truth_G_anc.edges())
    
    TP = len(model_output_G_anc_edge.intersection(ground_truth_G_anc_edge))
    FP = len(model_output_G_anc_edge) - TP
    FN = len(ground_truth_G_anc_edge) - TP
    
    if len(model_output_G_anc_edge) == 0:
        return 0, 0, 0, [TP, FP, FN]
    
    coverage_r = len(model_output_G_anc_edge.intersection(ground_truth_G_anc_edge)) / len(ground_truth_G_anc_edge)
    coverage_p = len(model_output_G_anc_edge.intersection(ground_truth_G_anc_edge)) / len(model_output_G_anc_edge)
    try:
        coverage_f1 = 2 * coverage_r * coverage_p / (coverage_r + coverage_p)
    except ZeroDivisionError:
        coverage_f1 = 0
        
    return coverage_p ,coverage_r, coverage_f1, [TP, FP, FN]


def construct_taxonomy(root, entities, relations):
    taxonomy = []
    prefix_dict = {}
    for entity in entities:
        prefix_dict[entity] = None
        
    level_idx_dict = {}
    entities = list(set(entities))
    #relation = list(set(relations))
    
    level_count_dict = {}
    tree = {}
    
    if len(entities) == 1:
        return '1. ' + str(root) + '\n'
    
    if relations == 0:
        raise NotImplementedError
    
    entity_in_relation = set()
    for relation in relations:
        entity_in_relation.add(relation[0])
        entity_in_relation.add(relation[1])

    if root not in entities:
        raise ValueError('The root is not in the entities')
    
    entities = set(entities)
    if len(entity_in_relation) != len(entities):
        print(relations)
        print(len(entity_in_relation), len(entities))
        # entity in relation that not in entities:
        print(entity_in_relation - entities)
        raise ValueError('Some entities are not in the relations')
    
    entity_in_relation = list(entity_in_relation)
    entities = entity_in_relation
    entity_level_dict = {entity:None for entity in entities}
    


    count = 0
    while True:
        add_in_this_iter = False
        for relation in relations:
            par, child = relation
            if par == root:
                prefix_dict[par] = '1.'
                if entity_level_dict[par] is None:
                    entity_level_dict[par] = 1
                    add_in_this_iter = True
            if entity_level_dict[par] is None:
                continue
            if entity_level_dict[child] is None:
                entity_level_dict[child] = entity_level_dict[par] + 1
                add_in_this_iter = True
                

        if add_in_this_iter == False:
            # remove all key:value paris with value is None
            entity_level_dict = {k:v for k,v in entity_level_dict.items() if v is not None}
            # remove all relation that has an entity not in entity_level_dict
            relations = [relation for relation in relations if relation[0] in entity_level_dict and relation[1] in entity_level_dict]
            # re-construct entity_in_relation
            entity_in_relation = set()
            for relation in relations:
                entity_in_relation.add(relation[0])
                entity_in_relation.add(relation[1])
            entity_in_relation = list(entity_in_relation)
            # re-construct entities
            entities = entity_in_relation
        
        if None not in entity_level_dict.values():
            break

    # sort relations by level
    sorted_relations = []
    for relation in relations:
        par, child = relation
        sorted_relations.append((entity_level_dict[par], relation))
    sorted_relations = sorted(sorted_relations, key=lambda x:x[0])

    relations = [relation for level, relation in sorted_relations]
        
    for relation in relations:
        par, child = relation    
        tree[par] = tree.get(par, {})
        tree[par][child] = tree[par].get(child, {})
        if prefix_dict[child] is None:
            level_idx_dict[prefix_dict[par]] = level_idx_dict.get(prefix_dict[par], 0) + 1
            level_count_dict[entity_level_dict[child]] = level_count_dict.get(entity_level_dict[child], 0) + 1
            if not prefix_dict[par].endswith('.'):
                prefix_dict[child] = prefix_dict[par] + '.' + str(level_idx_dict[prefix_dict[par]])
            else:
                prefix_dict[child] = prefix_dict[par] + str(level_idx_dict[prefix_dict[par]])
            
    '''
    # sort prefix_dict by prefix, notice that 1.2 should be before 1.10
    prefix2entity_dict = {}
    for entity, prefix in prefix_dict.items():
        prefix2entity_dict[prefix] = entity
        
    prefix_dict = dict(sorted(prefix_dict.items(), key=lambda item: item[1]))
    
    for entity, prefix in prefix_dict.items():
        taxonomy.append(prefix + ' ' + entity)
    '''
        
    # traverse tree, save taxonomy into string
    taxo_str_list = []
    def traverse_tree(tree, prefix_dict, root, prefix, taxo_str_list):
        if root not in tree:
            return
        for child in tree[root]:
            if len(taxo_str_list) == len(prefix_dict) - 1:
                return
            prefix = prefix_dict[child]
            #print(prefix, child)
            taxo_str_list.append(prefix + ' ' + child)
            traverse_tree(tree, prefix_dict, child, prefix, taxo_str_list)

    traverse_tree(tree, prefix_dict, root, '1.', taxo_str_list)    
    taxo_str = '1. ' + root + '\n' + "\n".join(taxo_str_list)
        
    return taxo_str

def call_api (client, prompt_list, save_path, model, times = 1, check = False, new_prompt = False):
    '''
    call the API to generate the response
    prompt_list: the prompt list
    save_path: the path to save the generated response
    model: the model name
    times: the number of times to generate the response
    
    save the response into save_path + 'model_response.npy' and save_path + 'model_response.json'
    '''
    if model == 'gpt-3.5-turbo-16k':
        max_tokens = 8000
    elif model == 'gpt-4-1106-preview':
        max_tokens = 4000
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    try:
        model_response = np.load(save_path + 'model_response.npy').tolist()
    except FileNotFoundError:
        model_response = None
    else:
        model_response = None

    if model_response is None:
        model_response = []
        count = 0
    else:
        count = len(model_response)
    for j, prompt in enumerate(tqdm(prompt_list)):
        if j < count:
            continue
        five_times = []
        for i in range(times):
            response = None
            check_time = 0
            while response is None:
                try:
                    response = client.chat.completions.create(
                        model = model,
                        messages=[
                                    {"role": "system", "content": 'You are an expert in constructing a taxonomy from a list of concepts.'},
                                    {"role": "user", "content": prompt}
                                ],
                        max_tokens = max_tokens,
                        
                    )
                except Exception as e:
                    print('Error: ', e)
                    time.sleep(20)
                    continue
                
                response_text = response.choices[0].message.content
                if check:
                    if check_time >= 5:
                        break
                    if new_prompt:
                        each_edges = [each.strip().strip(';').split(' is a subtopic of ') for each in response_text.split(';') if ' is a subtopic of ' in each]
                        each_edges = [each for each in each_edges if len(each) == 2]
                        each_edges = [(par.lower(), chi.lower()) for par, chi in each_edges]
                        each_entities_set = set()
                        for each in each_edges:
                            each_entities_set.add(each[0])
                            each_entities_set.add(each[1])
                    else:
                        each_edges, each_entities_set = phrase_taxo(response_text)
                    if len(each_entities_set) == 0:
                        print('re-run!')
                        response = None
                        check_time += 1
            five_times.append(response_text)
        model_response.append(five_times)

    np.save(save_path + 'model_response.npy', model_response)
    with open(save_path + 'model_response.json', 'w') as f:
        json.dump(model_response, f)
        
def find_roots(relation_list, entity_list):
    '''
    find the roots of the taxonomy
    relation_list: the relation list
    entity_list: the entity list
    
    return: the roots of the taxonomy
    '''
    in_degree = {entity: 0 for entity in entity_list}
    for relation in relation_list:
        in_degree[relation[1]] += 1
    roots = []
    for entity in entity_list:
        if in_degree[entity] == 0:
            roots.append(entity)
    return roots
