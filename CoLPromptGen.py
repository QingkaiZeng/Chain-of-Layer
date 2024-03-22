import os
import json
from utils import construct_taxonomy

def gen_ChainofLayers_prompt(taxo_name, demo_path, numofExamples = 5):
    
    with open(demo_path + taxo_name + '/demo.json', 'r') as f:
        demos = f.readlines()
        demos = [json.loads(demo) for demo in demos][0:numofExamples]

    instruction_line = "Build a taxonomy whose root concept is <root> with the given list of entities. The format of the generated taxonomy is: 1. Parent Concept 1.1 Child Concept. Do not change any entity names when building the taxonomy. Do not add any comments. There should be one and only one root node of the taxonomy. All entities in the entity list must appear in the taxonomy and don't add any entities that are not in the entity list.\n"

    entity_list_line = "Entity list: <entity_list>\n"

    Stepbysetp = "Let's do it step by step.\n"

    Firststep = "First, the entity in the first level of the taxonomy is <root>.\nThe current taxonomy is:\n"

    checkno = "Check: Is the remaining entity list empty?\nAnswer: No.\n"
    checkyes = "Check: Is the remaining entity list empty?\nAnswer: Yes.\nThe taxonomy is complete.\n"

    currenttaxo = "The current taxonomy is:\n"

    Nstep = "Then, let's find all the <N>-level entities from the remaining entity list.\n"

    def construct_taxonomy_with_depth(root, entity_list, relation_list, current_depth = 1):
        taxo = construct_taxonomy(root, entity_list, relation_list)
        taxo_lines = taxo.split('\n')
        level_lines_dict = {}
        level_entity_dict = {}
        for each in taxo_lines:
            idx = each.split(' ')[0]
            depth = len([e for e in idx.split('.') if e != ''])
            entity = ' '.join(each.split(' ')[1:])
            level_lines_dict.setdefault(depth, []).append((idx, entity))
            level_entity_dict.setdefault(depth, []).append(entity)
            
        entity_level_dict = {}
        for level in level_entity_dict:
            for entity in level_entity_dict[level]:
                entity_level_dict[entity] = level
                
        relation_list = [r for r in relation_list if r[0] in entity_level_dict and r[1] in entity_level_dict]

        level_relation_dict = {}
        for h,t in relation_list:
            h_depth = entity_level_dict[h]
            level_relation_dict[h_depth+1] = level_relation_dict.get(h_depth+1, []) + [[h,t]]
            
        if current_depth == 1:
            return ' '.join(level_lines_dict[1][0])
        cumulate_entity_list = []
        cumulate_relation_list = []
        for i in range(current_depth):
            cumulate_entity_list += level_entity_dict[i+1]
            cumulate_relation_list += level_relation_dict.get(i+1, [])
        
        taxo = construct_taxonomy(root, cumulate_entity_list, cumulate_relation_list)
        return taxo

    def cul_max_depth(root, entity_list, relation_list):
        taxo = construct_taxonomy(root, entity_list, relation_list)
        taxo_lines = taxo.split('\n')
        level_lines_dict = {}
        level_entity_dict = {}
        for each in taxo_lines:
            idx = each.split(' ')[0]
            depth = len([e for e in idx.split('.') if e != ''])
            entity = ' '.join(each.split(' ')[1:])
            level_lines_dict.setdefault(depth, []).append((idx, entity))
            level_entity_dict.setdefault(depth, []).append(entity)
            
        return len(level_lines_dict)
    
    complete_prompt = ''
    for j in range(len(demos)):
        root = demos[j]['root']
        entity_list = demos[j]['entity_list']
        relation_list = demos[j]['relation_list']
        max_depth = cul_max_depth(root, entity_list, relation_list)

        for i in range(max_depth):
            current_level = i+1
            if i+1 == 1:
                prompt = instruction_line + '\n' + entity_list_line + '\n' + Stepbysetp + Firststep
                #print(prompt)
            else:
                prompt += checkno + '\n'
                prompt += Nstep.replace('<N>', str(current_level))
                #print(prompt)
            taxo = construct_taxonomy_with_depth(root, entity_list, relation_list, current_level)
            prompt += taxo + '\n\n'
        prompt += checkyes + '\n'
        prompt = prompt.replace('<root>', root).replace('<entity_list>', str(entity_list))
        complete_prompt += prompt
        
    complete_prompt += instruction_line + '\n' + entity_list_line + '\n' + Stepbysetp + '\n'
        
    return complete_prompt


def gen_ChainofLayers_prompt_iterative(taxo_name, demo_path, numofExamples = 5):

    def construct_taxonomy_with_depth(root, entity_list, relation_list, current_depth = 1):
        taxo = construct_taxonomy(root, entity_list, relation_list)
        taxo_lines = taxo.split('\n')
        level_lines_dict = {}
        level_entity_dict = {}
        for each in taxo_lines:
            idx = each.split(' ')[0]
            depth = len([e for e in idx.split('.') if e != ''])
            entity = ' '.join(each.split(' ')[1:])
            level_lines_dict.setdefault(depth, []).append((idx, entity))
            level_entity_dict.setdefault(depth, []).append(entity)
            
        entity_level_dict = {}
        for level in level_entity_dict:
            for entity in level_entity_dict[level]:
                entity_level_dict[entity] = level
                
        relation_list = [r for r in relation_list if r[0] in entity_level_dict and r[1] in entity_level_dict]

        level_relation_dict = {}
        for h,t in relation_list:
            h_depth = entity_level_dict[h]
            level_relation_dict[h_depth+1] = level_relation_dict.get(h_depth+1, []) + [[h,t]]
            
        if current_depth == 1:
            return ' '.join(level_lines_dict[1][0])
        cumulate_entity_list = []
        cumulate_relation_list = []
        for i in range(current_depth):
            cumulate_entity_list += level_entity_dict[i+1]
            cumulate_relation_list += level_relation_dict.get(i+1, [])
        
        taxo = construct_taxonomy(root, cumulate_entity_list, cumulate_relation_list)
        return taxo

    def cul_max_depth(root, entity_list, relation_list):
        taxo = construct_taxonomy(root, entity_list, relation_list)
        taxo_lines = taxo.split('\n')
        level_lines_dict = {}
        level_entity_dict = {}
        for each in taxo_lines:
            idx = each.split(' ')[0]
            depth = len([e for e in idx.split('.') if e != ''])
            entity = ' '.join(each.split(' ')[1:])
            level_lines_dict.setdefault(depth, []).append((idx, entity))
            level_entity_dict.setdefault(depth, []).append(entity)
            
        return len(level_lines_dict)
    
    
    messages = []
    messages.append({"role": "system", "content": 'You are an expert in constructing a taxonomy from a list of concepts.'})
    
    with open(demo_path + taxo_name + '/demo.json', 'r') as f:
        demos = f.readlines()
        demos = [json.loads(demo) for demo in demos][0:numofExamples]

    instruction_line = "Build a taxonomy whose root concept is <root> with the given list of entities. The format of the generated taxonomy is: 1. Parent Concept 1.1 Child Concept. Do not change any entity names when building the taxonomy. Do not add any comments. There should be one and only one root node of the taxonomy. All entities in the entity list must appear in the taxonomy and don't add any entities that are not in the entity list.\n"

    entity_list_line = "Entity list: <entity_list>\n"

    Stepbysetp = "Let's do it step by step.\n"

    Firststep = "First, the entity in the first level of the taxonomy is <root>.\nThe current taxonomy is:\n"

    checkno = "Check: Is the remaining entity list empty?\n"
    ansno = "Answer: No.\n"
    checkyes = "Check: Is the remaining entity list empty?\n"
    ansyes = "Answer: Yes.\nThe taxonomy is complete.\n"
    
    #currenttaxo = "Expand the taxonomy with the remaining entity list.\nThe current taxonomy is:\n"
    currenttaxo = "The current taxonomy is:\n"

    Nstep = "Then, let's find all the <N>-level entities from the remaining entity list.\n"
    
    role_user_temp = {"role": "user", "content": '<USER>'}
    role_assistant_temp = {"role": "assistant", "content": '<ASSISTANT>'}

    #complete_prompt = ''
    for j in range(len(demos)):
        root = demos[j]['root']
        entity_list = demos[j]['entity_list']
        relation_list = demos[j]['relation_list']
        max_depth = cul_max_depth(root, entity_list, relation_list)
        
        
        for i in range(max_depth):
            current_level = i+1
            if current_level == 1:
                user_prompt = instruction_line.replace('<root>', root) + '\n' + entity_list_line.replace('<entity_list>', str(entity_list)) + '\n' + Stepbysetp
                #print(root, entity_list, relation_list)
                #print(user_prompt)
                role_user_current = role_user_temp.copy()
                role_user_current['content'] = user_prompt
                messages.append(role_user_current)
                
                assistant_response = Firststep.replace('<root>', root)
                

                #prompt = instruction_line + '\n' + entity_list_line + '\n' + Stepbysetp + Firststep
                #print(prompt)
            else:
                #prompt += checkno + '\n'
                user_prompt = Nstep.replace('<N>', str(current_level))
                role_user_current = role_user_temp.copy()
                role_user_current['content'] = user_prompt
                messages.append(role_user_current)
                #prompt += Nstep.replace('<N>', str(current_level))
                #print(prompt)
                
            taxo = construct_taxonomy_with_depth(root, entity_list, relation_list, current_level)
            
            if i+1 == 1:
                assistant_response += taxo + '\n\n'
                
            else:
                assistant_response = currenttaxo + '\n' + taxo + '\n\n'
            if i < max_depth:
                role_assistant_current = role_assistant_temp.copy()
                role_assistant_current['content'] = assistant_response
                messages.append(role_assistant_current)
                if i < max_depth - 1:
                    user_prompt = checkno
                    role_user_current = role_user_temp.copy()
                    role_user_current['content'] = user_prompt
                    messages.append(role_user_current)
                    
                    
                    assistant_response = ansno + '\n'
                    role_assistant_current = role_assistant_temp.copy()
                    role_assistant_current['content'] = assistant_response
                    messages.append(role_assistant_current)

            #prompt += taxo + '\n\n'
        
        user_prompt = checkyes
        role_user_current = role_user_temp.copy()
        role_user_current['content'] = user_prompt
        messages.append(role_user_current)
        
        assistant_response = ansyes + '\n'
        role_assistant_current = role_assistant_temp.copy()
        role_assistant_current['content'] = assistant_response
        messages.append(role_assistant_current)
        #prompt += checkyes + '\n'
        #prompt = prompt.replace('<root>', root).replace('<entity_list>', str(entity_list))
        #complete_prompt += prompt
        
    user_prompt = instruction_line + '\n' + entity_list_line + '\n' + Stepbysetp + '\n'
    role_user_current = role_user_temp.copy()
    role_user_current['content'] = user_prompt
    messages.append(role_user_current)
    
    #complete_prompt += instruction_line + '\n' + entity_list_line + '\n' + Stepbysetp + '\n'
        
    return messages