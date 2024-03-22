import time
from tqdm import tqdm
from utils import *
import numpy as np
import os
import json
import random
import argparse
from openai import OpenAI

def gen_promt_template(mode = 0):
    '''
    Generate the prompt template for the demo.
    
    return: the prompt template.
    '''
    if mode == 0:
        prompt_temp = "Build a taxonomy whose root concept is <root> with the given list of entities. The format of generated taxonomy is: 1. Parent Concept 1.1 Child Concept. Do not change any entity names when building the taxonomy. Do not add any comment. There should be one and only one root node of the taxonomy. All entities in the entity list must appear in the taxonomy and don't add any entities that are not in the entity list.\n"
    elif mode == 1:
        prompt_temp = 'Generate 5 different taxonomies about <root>. The format of the generated taxonomy is: 1. Parent Concept 1.1 Child Concept. Do not add any comments. There should be one and only one root node of each taxonomy.\n'
    else:
        raise NotImplementedError

    return prompt_temp

def run(client, taxo_name, taxo_path, model, save_path, numofExamples = 0, file_name = 'test.json', mode = 0):
    '''
    generate the demo. save the generated demo to save_path.
    
    taxo_name: the name of the taxonomy.
    taxo_path: the path of the taxonomy.
    model: the model name.
    save_path: the path to save the generated demo.
    numofExamples: the number of examples to generate.
    file_name: the name of the file of the generated demo.
    '''
    path = taxo_path + taxo_name + '/' + file_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path + taxo_name):
        os.makedirs(save_path + taxo_name)
    save_path = save_path + taxo_name + '/'

    with open(path, 'r') as f:
        subgraphs = f.readlines()
        subgraphs = [json.loads(line) for line in subgraphs]
        
    prompt_list = []
    
    for i, subgraph in enumerate(subgraphs):
        prompt_temp = gen_promt_template(mode = mode)
        entities = subgraph['entity_list']
        relations = subgraph['relation_list']
        root = subgraph['root']
        if numofExamples == 0:
            if mode == 0:
                for i in range(5):
                    random.shuffle(entities)
                    #numofSamples = random.randint(10, 15)
                    numofSamples = 5
                    sample_entities = entities[:numofSamples]
                    prompt_list.append(prompt_temp.replace('<root>', root) + "Entity List: " + str(sample_entities) + "\n" + "Taxonomy:\n")
            elif mode == 1:
                prompt_list.append(prompt_temp.replace('<root>', root))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
    call_api(client, prompt_list, save_path, model, times = 1, check = False)
    
    with open(save_path + 'prompt_list.json', 'w') as f:
        json.dump(prompt_list, f)
        
def split_subtree(relation_list, entity_list, roots):
    '''
    Split the taxonomy with more than one root into subtrees that each has only one root.
    
    relation_list: the list of relations in the taxonomy.
    entity_list: the list of entities in the taxonomy.
    roots: the list of roots in the taxonomy.
    
    return: the list of subtrees.
    '''
    subtree = {}
    for root in roots:
        subtree[root] = {'relation_list': [], 'entity_list': [root]}
        
    add_flag = {relation: True for relation in relation_list}
    
    while True:
        
        for relation in relation_list:
            if add_flag[relation]:
                if relation[0] in roots:
                    subtree[relation[0]]['relation_list'].append(relation)
                    subtree[relation[0]]['entity_list'].append(relation[1])
                    add_flag[relation] = False
                    
                for root in roots:
                    if relation[0] in subtree[root]['entity_list']:
                        subtree[root]['relation_list'].append(relation)
                        subtree[root]['entity_list'].append(relation[1])
                        add_flag[relation] = False
                        
                        break
        if sum(add_flag.values()) == 0:
            break
    
    for root in roots:
        subtree[root]['entity_list'] = list(set(subtree[root]['entity_list']))
        subtree[root]['relation_list'] = list(set(subtree[root]['relation_list']))
        
    return subtree

def pharse_model_response_to_json(model_response_path, save_path, taxo_name, file_name = 'test.json'):
    with open(model_response_path) as f:
        taxos = f.readlines()
    taxos = [json.loads(taxo) for taxo in taxos][0]
    
    save_path += taxo_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with open(save_path+file_name, 'w') as f:
        for taxo in taxos:
            relation_list, entity_list = phrase_taxo(taxo[0])
            roots = find_roots(relation_list, entity_list)
            subtree = split_subtree(relation_list, entity_list, roots)
            for root in roots:
                dict = {'root': root, 'relation_list': subtree[root]['relation_list'], 'entity_list': subtree[root]['entity_list']}
                f.write(json.dumps(dict) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--taxo_name', type=str, default='semeval_sci')
    parser.add_argument('--taxo_path', type=str, default='./dataset/processed/')
    parser.add_argument('--save_path_model_response', type=str, default='./demos/raw_demo_gen/')
    parser.add_argument('--save_path', type=str, default='./demos/demo_gen/')
    parser.add_argument('--model', type=str, default='gpt-4-1106-preview')
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--numofExamples', type=int, default=0)
    parser.add_argument('--openai_key', type=str)
    args = parser.parse_args()
    
    client = OpenAI(api_key=args.openai_key)
    taxo_name = args.taxo_name
    taxo_path = args.taxo_path
    model = args.model
    numofExamples = args.numofExamples
    save_path_model_response = args.save_path_model_response
    save_path = args.save_path
    mode = args.mode
    #print(taxo_name, taxo_path, model, numofExamples, save_path)
    run(client, taxo_name, taxo_path, model, save_path_model_response, numofExamples, mode = mode)
    model_response_path = save_path_model_response + taxo_name + '/model_response.json'
    pharse_model_response_to_json(model_response_path, save_path, taxo_name)