from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from torch.nn import functional as F
from tqdm import tqdm
import os
import json
import argparse

PROMPTS = {
    'p2a': "<term> is a <mask>",
    'p2b': "<term> is an <mask>",
    'p3a': " <term> is a kind of <mask>",
    'p3b': "<term> is a type of <mask>",
    'p3c': " <term> is an example of <mask>",
    'p4a': "<mask> such as <term>",
    'p4b': "A <mask> such as <term>",
    'p4c': "An <mask> such as <term>",
}

def get_terms(taxo_name):
    terms = []
    path = './dataset/processed/' + taxo_name + '/test.json'
    with open(path, 'r') as f:
        subgraphs = f.readlines()
        subgraphs = [json.loads(subgraph) for subgraph in subgraphs]
        for subgraph in subgraphs:
            terms.append(subgraph['entity_list'])
    return terms

def get_prompt(prompt, focus, mask):
    prompt = PROMPTS[prompt]
    prompt = prompt.replace('<term>', focus)
    prompt = prompt.replace('<mask>', mask)
    return prompt

def load_pretrained(model_path, device):
    """
    Load pretrained HuggingFace tokenizer and model.
    :param model_name: model checkpoint
    :param device: CPU / CUDA device
    :return: tokenizer and model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path, return_dict=True)
    model.to(device)
    model.eval()
    return tokenizer, model

def get_target_vocab(tokenizer, terms):
    ids = []
    target_terms = []
    for term in terms:
        tokenized = tokenizer.tokenize(term)
        if len(tokenized) == 1:
            id = tokenizer.convert_tokens_to_ids(tokenized)
            target_terms.append(tokenized[0])
            ids.append(id)
    ids = torch.tensor(ids).squeeze(1)

    # fill corresponding ids with 1, 0 for remaining
    target_vocab = torch.zeros(tokenizer.vocab_size)
    target_vocab.index_fill_(0, ids, 1)

    return target_vocab


def calculate(terms, model_name, model, tokenizer, device, template):
    vocab = tokenizer.get_vocab()
    results = {}
    mask_token = tokenizer.mask_token
    # Generate masked sentence templates
    
    template = template.replace('<mask>', mask_token)
    '''
    if 'bert' in model_name and 'roberta' not in model_name:
        template = template.replace('<mask>', '[MASK]')
    '''
    with torch.no_grad():
        for main_term in terms:  # Main term to calculate rank for
            inputs = tokenizer(template.replace("<term>", main_term), return_tensors="pt").to(device)
            mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
            
            # Get logits at the masked token position
            outputs = model(**inputs)
            prediction_logits = outputs.logits.squeeze(0)[mask_token_index.item(), :]
            
            # Sort the logits to get the rankings
            probs = torch.softmax(prediction_logits, dim=-1)
            ranking = torch.argsort(probs, descending=True)

            term_ranks = {}

            for term in terms:
                if term == main_term:  # Skip the main term
                    continue

                token_ids = tokenizer.encode(term, add_special_tokens=False)
                token_ranks = []

                for token_id in token_ids:
                    mask_rank = (ranking == token_id).nonzero(as_tuple=True)
                    if mask_rank[0].size(0) > 0:
                        token_rank = mask_rank[0][0].item() + 1
                    else:
                        token_rank = len(ranking)  # Assign the lowest rank if not found
                    token_ranks.append(token_rank)

                average_rank = sum(token_ranks) / len(token_ranks)
                term_ranks[term] = average_rank

            results[main_term] = term_ranks
            
    return results

    # Print the results
    for term, rank_dict in results.items():
        sorted_ranks = sorted(rank_dict.items(), key=lambda x: x[1])
        print(f"Term '{term}' ranks:")
        for other_term, rank in sorted_ranks:
            print(f"{other_term}: {rank}")
        print()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--taxo_name', type=str, default='semeval_sci')
    parser.add_argument('--save_path', type=str, default='./filter/')
    parser.add_argument('--model_path', type=str, default='allenai/scibert_scivocab_uncased')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model_path = args.model_name
    save_path = args.save_path
    taxo_name = args.taxo_name
    tokenizer, model = load_pretrained(model_path, device)
    print('Start to calculate scores for ' + taxo_name + ' using ' + model_path + ' model')

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if '/' in model_path:
        save_path = save_path + model_path.split('/')[1] + '/'
    else:
        save_path = save_path + model_path + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = save_path + taxo_name + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    terms_list = get_terms(taxo_name)
    fw = open(save_path + 'scores.json', 'w')
    for terms in tqdm(terms_list):
        ranks = {}
        scores = {}
        for name, template in PROMPTS.items():
            prompt = get_prompt(name, terms[0], '<mask>')
            ranks[template] = calculate(terms, model_path, model, tokenizer, device=device, template=prompt)
        
        for name, template in PROMPTS.items():
            template_rank = ranks[template]
            for main_term, other_terms_rank in template_rank.items():
                scores[main_term] = {}
                for other_term, rank in other_terms_rank.items():
                    scores[main_term][other_term] = scores[main_term].get(other_term, 0) + 1/rank
        fw.write(json.dumps(scores) + '\n')