import json
import os
import pickle
import re
import math

import nltk
import torch
from tqdm import tqdm
import shutil
import numpy as np
from collections import Counter

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
        
    cached_dictionaries_orig = os.path.join(clevr_dir, 'questions', 'CLEVR_built_dictionaries.pkl')
    cached_dictionaries = os.path.join(clevr_dir, 'CLEVR-Humans', 'CLEVR_built_dictionaries.pkl')

    if os.path.exists(cached_dictionaries):
        print('==> using cached dictionaries: {}'.format(cached_dictionaries))
        with open(cached_dictionaries, 'rb') as f:
            return pickle.load(f)
            
    with open(cached_dictionaries_orig, 'rb') as f:
        quest_to_ix, answ_to_ix, answ_ix_to_class = pickle.load(f)

    # quest_to_ix = {}
    # answ_to_ix = {}
    # answ_ix_to_class = {}
    json_train_filename = os.path.join(clevr_dir, 'CLEVR-Humans', 'CLEVR-Humans-train.json')
    #load all words from all training data
    counter = Counter()

    with open(json_train_filename, "r") as f:
        questions = json.load(f)['questions']
        for q in tqdm(questions):
            question_tokens = nltk.tokenize.word_tokenize(q['question'])
            answer = q['answer']
            #pdb.set_trace()
            counter.update(question_tokens)
            '''
            for word in question:
                if word not in quest_to_ix:
                    quest_to_ix[word] = len(quest_to_ix)+1 #one based indexing; zero is reserved for padding
            '''
    threshold = 4
    words = [word for word, cnt in counter.items() if cnt >= threshold and word not in quest_to_ix]

    i = max(quest_to_ix.values())+1

    for w in words:
        quest_to_ix[w] = i
        i += 1

    ret = (quest_to_ix, answ_to_ix, answ_ix_to_class)    
    with open(cached_dictionaries, 'wb') as f:
        pickle.dump(ret, f)

    return ret


def to_dictionary_indexes(dictionary, sentence):
    """
    Outputs indexes of the dictionary corresponding to the words in the sequence.
    Case insensitive.
    """
    split = nltk.tokenize.word_tokenize(sentence)
    idxs = torch.LongTensor([dictionary[w] if w in dictionary else 0 for w in split])
    return idxs

def collate_data(batch):
    images, lengths = [], []
    idxs = []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    for i, b in enumerate(sort_by_len):
        image, question, length, q_idx = b
        idxs.append(q_idx)
        images.append(torch.from_numpy(image))
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)

    return torch.stack(images), torch.from_numpy(questions), \
        lengths, idxs

def set_optimizer_lr(optimizer, lr):
    # callback to set the learning rate in an optimizer, without rebuilding the whole optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
    
def get_sgdr_lr(period, epoch_id, eta_min, eta_max):
    '''
    Tmax: period
    Tcurr: batch_idx
    '''

    radians = math.pi*(epoch_id/period)
    return eta_min + 0.5 * (eta_min + eta_max) * (1.0 + math.cos(radians))

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
