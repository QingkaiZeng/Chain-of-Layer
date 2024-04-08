import time
from tqdm import tqdm
from utils import *
import numpy as np
import os
import json
import random
from openai import OpenAI
from metric import eval
import argparse
from CoLPromptGen import gen_ChainofLayers_prompt, gen_ChainofLayers_prompt_iterative

random.seed(42)

def gen_promt_template(taxo_name, demo_path, demo_name = 'demo.json', numofExamples = 0, start = 0):
    '''
    generate prompt template for the given taxonomy
    taxo_name: the name of the taxonomy
    demo_path: the path of the demo file
    demo_name: the name of the demo file
    numofExamples: the number of incontext examples
    start: the start index of the incontext examples
    
    return: the prompt template
    '''
    demo_path = demo_path + taxo_name + '/'
    
    if numofExamples == 0:
        prompt_temp = "Build a taxonomy whose root concept is <root> with the given list of entities. The format of generated taxonomy is: 1. Parent Concept 1.1 Child Concept. Do not change any entity names when building the taxonomy. Do not add any comment. There should be one and only one root node of the taxonomy. All entities in the entity list must appear in the taxonomy and don't add any entities that are not in the entity list.\n"
    else:
        prompt_temp = "Build a taxonomy whose root concept is <root> with the given list of entities. The format of generated taxonomy is: 1. Parent Concept 1.1 Child Concept. Do not change any entity names when building the taxonomy. Do not add any comment. There should be one and only one root node of the taxonomy. All entities in the entity list must appear in the taxonomy and don't add any entities that are not in the entity list.\n"
        
        with open(demo_path + demo_name, 'r') as f:
            examples_subgraph = f.readlines()
            for i, line in enumerate(examples_subgraph):
                examples_subgraph[i] = json.loads(line)
        
        for i in range(start, start+numofExamples):
            instrcution_prompt = "Build a taxonomy whose root concept is <root> with the given list of entities. The format of generated taxonomy is: 1. Parent Concept 1.1 Child Concept. Do not change any entity names when building the taxonomy. Do not add any comment. There should be one and only one root node of the taxonomy. All entities in the entity list must appear in the taxonomy and don't add any entities that are not in the entity list.\n"
            
            root = examples_subgraph[i]['root']
            entities = examples_subgraph[i]['entity_list']
            relations = examples_subgraph[i]['relation_list']
            example_taxo = construct_taxonomy(root, entities, relations)
            prompt_temp = prompt_temp.replace('<root>', root)
            prompt_temp += "Entity List: " + str(entities) + "\n" + "Taxonomy:\n" + example_taxo + "\n"
            prompt_temp += '\n'
            prompt_temp += instrcution_prompt
        
    return prompt_temp

def gen_promt_template_new(taxo_name, demo_path, demo_name = 'demo.json', numofExamples = 0, start = 0):
    '''
    generate prompt template for the given taxonomy
    taxo_name: the name of the taxonomy
    demo_path: the path of the demo file
    demo_name: the name of the demo file
    numofExamples: the number of incontext examples
    start: the start index of the incontext examples
    
    return: the prompt template
    '''
    demo_path = demo_path + taxo_name + '/'
    
    if numofExamples == 0:
        prompt_temp = "You are an expert constructing a taxonomy from a list of concepts. Given a list of concepts, construct a taxonomy by creating a list of their parent-child relationships. Use the format of 'child is a subtopic of parent'\n\n\n"
    else:
        prompt_temp = "You are an expert constructing a taxonomy from a list of concepts. Given a list of concepts, construct a taxonomy by creating a list of their parent-child relationships. Use the format of 'child is a subtopic of parent'\n\n\n"
        
        with open(demo_path + demo_name, 'r') as f:
            examples_subgraph = f.readlines()
            for i, line in enumerate(examples_subgraph):
                examples_subgraph[i] = json.loads(line)
        
        for i in range(start, start+numofExamples):
            root = examples_subgraph[i]['root']
            entities = examples_subgraph[i]['entity_list']
            relations = examples_subgraph[i]['relation_list']
            concepts = '; '.join(entities)
            relationships = '; '.join([e2 + ' is a subtopic of ' + e1 for e1, e2 in relations])

            prompt_temp += "Concepts: " + concepts + "\n" + "Relationships: " + relationships + "\n"
            prompt_temp += '\n\n'
        
    return prompt_temp

def call_api_interative (client, messages, model, check = False):
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
    
    multi_times_response = []
    multi_times_message = []

    response = None
    while response is None:
        try:
            response = client.chat.completions.create(
                model = model,
                messages = messages,
                max_tokens = max_tokens,
                
            )
        except Exception as e:
            print('Error: ', e)
            time.sleep(20)
            continue
        
        response_text = response.choices[0].message.content
        
        if check:
            each_edges, each_entities_set = phrase_taxo(response_text)
            if len(each_entities_set) == 0:
                response = None

    return response_text, response

def add_response(response_text, messages):
    role_assistant = {"role": "assistant", "content": response_text}
    messages.append(role_assistant)
    return messages

def cal_hit_at_n(edge, filter_scores, n):
    # edge: [parent_term, child_term]
    # filter_scores: {child_term: {parent_term: score, ...}, ...}
    # n: top n
    # return: 1 or 0
    child_term = edge[1]
    parent_term = edge[0]
    if child_term not in filter_scores:
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

def filter(response_text, messages, root, gt_entities_list, filter_scores = None, filter_topk = None, filter_mode = None):
    if filter_mode == 'lm_score_ensemble':
        if filter_scores is None:
            raise ValueError('filter_scores is None')
        if filter_topk is None:
            raise ValueError('filter_topk is None')
    gt_entities_set = set(gt_entities_list)
    revised_gt_entities_set = set()
    for gt_entity in list(gt_entities_set):
        revised_gt_entities_set.add(gt_entity.lower())
    
    gt_entities_set = revised_gt_entities_set
    
    each_edges, each_entities_set = phrase_taxo(response_text)
    revised_entities_set = set()
    revised_edges = []
    #print('root: ', root)
    #print('gt_entities_set: ', gt_entities_set)
    #print('each_entities_set: ', each_entities_set)
    if len(each_entities_set) == 1:
        messages = add_response(response_text, messages)
        return messages
    
    else:
        for edge in each_edges:
            if edge[0].lower() in gt_entities_set and edge[1].lower() in gt_entities_set:
                revised_edges.append(edge)
            else:
                continue
        if filter_mode is None:
            for edge in revised_edges:
                revised_entities_set.add(edge[0].lower())
                revised_entities_set.add(edge[1].lower())
            revised_edges = set(revised_edges)
        
        elif filter_mode == 'lm_score_ensemble':
            filtered_edges = []
            for edge in revised_edges:
                cal_hit_at_n_result = cal_hit_at_n(edge, filter_scores, filter_topk)
                if cal_hit_at_n_result == 1:
                    filtered_edges.append(edge)
                else:
                    continue
            
            for edge in filtered_edges:
                revised_entities_set.add(edge[0].lower())
                revised_entities_set.add(edge[1].lower())
            revised_edges = set(filtered_edges)
                
        if len(revised_edges) == 0:
            revised_entities_set.add(root.lower())
        
        if root.lower() not in revised_entities_set:
            raise ValueError('root is not in revised_entities_set')
                
        #print('revised_entities_set: ', revised_entities_set)
        #print('revised_edges: ', revised_edges)
        currenttaxo = "The current taxonomy is:\n"
        taxo = construct_taxonomy(root.lower(), list(revised_entities_set), list(revised_edges))

        #print(currenttaxo + taxo)
        messages = add_response(currenttaxo + taxo, messages)
        return messages
    
def taxo_gen(client, messages_list, subgraphs, save_path, model, times = 1, check = False):
    '''
    call the API to generate the response
    prompt_list: the prompt list
    save_path: the path to save the generated response
    model: the model name
    times: the number of times to generate the response
    
    save the response into save_path + 'model_response.npy' and save_path + 'model_response.json'
    '''
    check_empty = "Check: Is the remaining entity list empty?\n"
    Nstep = "Then, let's find all the <N>-level entities from the remaining entity list.\n"
    
    if os.path.exists(save_path):
        if 'model_response.json' in os.listdir(save_path):
            with open(save_path + 'model_response.json', 'r') as f:
                model_response = json.load(f)
        else:
            model_response = []
    else:
        model_response = []
    
    complete_count = len(model_response)
    print('complete_count: ', complete_count)
    
    for j, messages in enumerate(tqdm(messages_list)):
        if j < complete_count:
            continue
        multi_times_response = []
        multi_times_message = []
        for i in range(times):
            N = 1
            iter_times = 0
            while True and iter_times < 5:
                #print(iter_times)
                response_text, response = call_api_interative(client, messages, model, check)
                print(response_text)
                if 'The taxonomy is complete.' in response_text:
                    messages = add_response(response_text, messages)
                    break
                
                if check_empty in messages[-1]['content']:
                    messages = add_response(response_text, messages)
                    N += 1
                    role_user = {"role": "user", "content": Nstep.replace('<N>', str(N))}
                    messages.append(role_user)
                else:
                    each_edges, each_entities_set = phrase_taxo(response_text)
                    if len(each_entities_set) == 0:
                        iter_times += 1
                        continue
                    messages = filter(response_text, messages, subgraphs[j]['root'], subgraphs[j]['entity_list'])
                    role_user = {"role": "user", "content": check_empty}
                    messages.append(role_user)
                    iter_times += 1
        
            multi_times_response.append(response_text)
            multi_times_message.append(messages)
        model_response.append(multi_times_message)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(save_path + 'model_response.json', 'w') as f:
            json.dump(model_response, f)
    
    #np.save(save_path + 'model_response.npy', model_response)
    with open(save_path + 'model_response.json', 'w') as f:
        json.dump(model_response, f)

def run(client, taxo_name, taxo_path, model, save_path_model_response, numofExamples = 0, file_name = 'test.json', new_prompt = False, ChainofLayers = False, iteratively = False, filter_mode = None, filter_topk = None, filter_scores_list = None):
    '''
    generate the prompt list and ground truth list for the given taxonomy, then call the API to generate the response
    
    taxo_name: the name of the taxonomy
    taxo_path: the path of the taxonomy
    model: the model name
    save_path_model_response: the path to save the generated taxonomy
    numofExamples: the number of incontext examples
    file_name: the name of the file that contains the subgraphs
    '''
    save_path = save_path_model_response
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if ChainofLayers and iteratively:
        if filter_topk is not None:
            save_path = f'{save_path}{taxo_name}_top{filter_topk}/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        else:
            save_path = save_path + taxo_name + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        
    save_path = save_path + taxo_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = save_path + model + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = save_path + str(numofExamples) + 'shots/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    path = taxo_path + taxo_name + '/' + file_name
    with open(path, 'r') as f:
        subgraphs = f.readlines()
        subgraphs = [json.loads(line) for line in subgraphs]
    
    prompt_list = []
    ground_truth_list = []
    
    for i, subgraph in enumerate(subgraphs):
        if new_prompt:
            prompt_temp = gen_promt_template_new(taxo_name, demo_path, numofExamples = numofExamples)
        else:
            if not ChainofLayers and not iteratively: # direct
                prompt_temp = gen_promt_template(taxo_name, demo_path, numofExamples = numofExamples)
            elif ChainofLayers and iteratively: # CoL-iterative
                prompt_temp = gen_ChainofLayers_prompt_iterative(taxo_name, demo_path, numofExamples = numofExamples)
            elif not ChainofLayers and iteratively:
                raise NotImplementedError
            else: # ChainofLayers and not iteratively
                prompt_temp = gen_ChainofLayers_prompt(taxo_name, demo_path, numofExamples = numofExamples)
        entities = subgraph['entity_list']
        random.shuffle(entities)
        relations = subgraph['relation_list']
        root = subgraph['root']
        if new_prompt:
            prompt_list.append(prompt_temp + "Concepts: " + '; '.join(entities) + "\n" + "Relationships: ")
        else:
            if not ChainofLayers and not iteratively: # direct
                prompt_list.append(prompt_temp.replace('<root>', root) + "Entity List: " + str(entities) + "\n" + "Taxonomy:\n")
            elif ChainofLayers and iteratively: # CoL-iterative
                prompt_temp_current = prompt_temp.copy()
                prompt_temp_current[-1] = prompt_temp_current[-1].copy()
                prompt_temp_current[-1]['content'] = prompt_temp_current[-1]['content'].replace('<root>', root).replace('<entity_list>', str(entities))
                prompt_list.append(prompt_temp_current)
            elif not ChainofLayers and iteratively:
                raise NotImplementedError
            else: # ChainofLayers and not iteratively
                prompt_list.append(prompt_temp.replace('<root>', root).replace('<entity_list>', str(entities)))
        ground_truth_list.append([(e1, e2) for e1, e2 in relations])
        
    if new_prompt:
        print('new_prompt')
        call_api(client, prompt_list, save_path, model, times = 1, check = False, new_prompt = True)
    else:
        if ChainofLayers and iteratively: # CoL-iterative
            taxo_gen(client, prompt_list, subgraphs, save_path, model, times = 1, check = False)
        else:
            call_api(client, prompt_list, save_path, model, times = 1, check = False)
    with open(save_path + 'prompt_list.json', 'w') as f:
        json.dump(prompt_list, f)
    with open(save_path + 'ground_truth_list.json', 'w') as f:
        json.dump(ground_truth_list, f)
    return ground_truth_list, prompt_list
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--taxo_name', type=str, default='semeval_sci')
    parser.add_argument('--taxo_path', type=str, default='./dataset/processed/')
    parser.add_argument('--demo_path', type=str, default='./demo_wordnet_train/')
    parser.add_argument('--model', type=str, default='gpt-4-1106-preview')
    parser.add_argument('--save_path_model_response', type=str, default='./results/taxo/')
    parser.add_argument('--numofExamples', type=int, default=0)
    parser.add_argument('--run', type=str, default='True')
    parser.add_argument('--new_prompt', type=str, default='False')
    parser.add_argument('--ChainofLayers', type=str, default='False')
    parser.add_argument('--iteratively', type=str, default='False')
    parser.add_argument('--filter_mode', type=str, default='None')
    parser.add_argument('--filter_topk', type=int, default=10)
    parser.add_argument('--filter_model', type=str, default='bert-base-uncased')
    parser.add_argument('--analysis', type=str, default='False')
    parser.add_argument('--openai_key', type=str)
    args = parser.parse_args()
    
    client = OpenAI(api_key=args.openai_key)
    if args.new_prompt == 'True':
        new_prompt = True
    else:
        new_prompt = False
        
    if args.ChainofLayers == 'True':
        ChainofLayers = True
    else:
        ChainofLayers = False
        
    if args.iteratively == 'True':
        iteratively = True
    else:
        iteratively = False
    
    if args.analysis == 'True':
        analysis = True
    else:
        analysis = False

    taxo_name = args.taxo_name
    taxo_path = args.taxo_path
    model = args.model
    numofExamples = args.numofExamples
    demo_path = args.demo_path
    save_path_model_response = args.save_path_model_response
        
    if args.filter_mode == 'None':
        filter_mode = None
        filter_topk = None
        filter_scores_list = None
    elif args.filter_mode == 'lm_score_ensemble':
        filter_mode = 'lm_score_ensemble'
        #mapping = {'wiki': 'wiki_downsample', 'dblp': 'dblp_sampled_downsample', 'semeval_sci': 'semeval_sci_downsample', 'wordnet': 'wordnet'}
        mapping = {'wiki_downsample': 'wiki_downsample', 'dblp_sampled_downsample': 'dblp_sampled_downsample', 'semeval_sci_downsample': 'semeval_sci_downsample', 'wordnet': 'wordnet'}
        filter_path = f'./filter/{args.filter_model}/{mapping[taxo_name]}/scores.json'
        filter_scores_list = open(filter_path, 'r').readlines()
        filter_scores_list = [json.loads(filter_scores) for filter_scores in filter_scores_list]
        filter_topk = args.filter_topk
    
    #print(taxo_name, taxo_path, model, numofExamples, save_path_model_response, demo_path)
    if args.run == 'True':
        run(client, taxo_name, taxo_path, model, save_path_model_response, numofExamples = numofExamples, new_prompt = new_prompt, ChainofLayers = ChainofLayers)
        eval(taxo_name, taxo_path, model, save_path_model_response, numofExamples = numofExamples, new_prompt = new_prompt, ChainofLayers = ChainofLayers, iteratively = iteratively, filter_mode = filter_mode, filter_topk = filter_topk, filter_scores_list = filter_scores_list)
    else:
        eval(taxo_name, taxo_path, model, save_path_model_response, numofExamples = numofExamples, new_prompt = new_prompt, ChainofLayers = ChainofLayers, iteratively = iteratively, filter_mode = filter_mode, filter_topk = filter_topk, filter_scores_list = filter_scores_list)