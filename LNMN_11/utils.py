import json
import os
import pickle
import re

import torch
from tqdm import tqdm
import shutil
import numpy as np
import subprocess

classes = {
            'number':['0','1','2','3','4','5','6','7','8','9','10'],
            'material':['rubber','metal'],
            'color':['cyan','blue','yellow','purple','red','green','gray','brown'],
            'shape':['sphere','cube','cylinder'],
            'size':['large','small'],
            'exist':['yes','no']
        }

def tokenize(sentence):
    # punctuation should be separated from the words
    s = re.sub('([.,;:!?()])', r' \1 ', sentence)
    s = re.sub('\s{2,}', ' ', s)

    # tokenize
    split = s.split()

    # normalize all words to lowercase
    lower = [w.lower() for w in split]
    return lower

def build_dictionaries(clevr_dir):

    def compute_class(answer):
        for name,values in classes.items():
            if answer in values:
                return name
        
        raise ValueError('Answer {} does not belong to a known class'.format(answer))
        
        
    cached_dictionaries = os.path.join(clevr_dir, 'questions', 'CLEVR_built_dictionaries.pkl')
    if os.path.exists(cached_dictionaries):
        print('==> using cached dictionaries: {}'.format(cached_dictionaries))
        with open(cached_dictionaries, 'rb') as f:
            return pickle.load(f)
            
    quest_to_ix = {}
    answ_to_ix = {}
    answ_ix_to_class = {}
    json_train_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_train_questions.json')
    #load all words from all training data
    with open(json_train_filename, "r") as f:
        questions = json.load(f)['questions']
        for q in tqdm(questions):
            question = tokenize(q['question'])
            answer = q['answer']
            #pdb.set_trace()
            for word in question:
                if word not in quest_to_ix:
                    quest_to_ix[word] = len(quest_to_ix)+1 #one based indexing; zero is reserved for padding
            
            a = answer.lower()
            if a not in answ_to_ix:
                    ix = len(answ_to_ix)+1
                    answ_to_ix[a] = ix
                    answ_ix_to_class[ix] = compute_class(a)

    ret = (quest_to_ix, answ_to_ix, answ_ix_to_class)    
    with open(cached_dictionaries, 'wb') as f:
        pickle.dump(ret, f)

    return ret


def to_dictionary_indexes(dictionary, sentence):
    """
    Outputs indexes of the dictionary corresponding to the words in the sequence.
    Case insensitive.
    """
    split = tokenize(sentence)
    idxs = torch.LongTensor([dictionary[w] for w in split])
    return idxs

def collate_data(batch):
    images, lengths, answers, families = [], [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    for i, b in enumerate(sort_by_len):
        image, question, length, answer, family = b
        images.append(torch.from_numpy(image))
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)
        families.append(family)

    return torch.stack(images), torch.from_numpy(questions), \
        torch.LongTensor(lengths), torch.LongTensor(answers), families

def collate_data_layout(batch):
    max_layout_len = 19
    images, lengths, answers, families = [], [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    padded_layouts = 8*torch.ones(batch_size, max_layout_len, dtype=torch.long)

    for i, b in enumerate(sort_by_len):
        image, question, length, answer, family, ques_layout = b
        images.append(torch.from_numpy(image))
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)
        families.append(family)

        padded_layouts[i, :len(ques_layout)] = torch.tensor(ques_layout, dtype=torch.long)

    return torch.stack(images), torch.from_numpy(questions), \
        torch.LongTensor(lengths), torch.LongTensor(answers), families, padded_layouts

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    print(gpu_memory_map)
    return gpu_memory_map

def get_positional_encoding(H, W):
    pe_dim = 128
    assert pe_dim % 4 == 0, 'pe_dim must be a multiply of 4 (h/w x sin/cos)'
    c_period = 10000. ** np.linspace(0., 1., pe_dim // 4)
    h_vec = np.tile(np.arange(0, H).reshape((H, 1, 1)), (1, W, 1)) / c_period
    w_vec = np.tile(np.arange(0, W).reshape((1, W, 1)), (H, 1, 1)) / c_period
    position_encoding = np.concatenate(
        (np.sin(h_vec), np.cos(h_vec), np.sin(w_vec), np.cos(w_vec)), axis=-1)
    position_encoding = np.transpose(position_encoding.reshape((H, W, pe_dim)), (2,0,1)).astype(np.float32)
    return position_encoding