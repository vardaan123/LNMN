import os
import pickle

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import h5py
import utils_clevr_humans
import torchvision
import json

from utils_clevr_humans import get_positional_encoding

class ClevrDataset(Dataset):
    def __init__(self, clevr_dir, split, dictionaries, features_dir='data'):
        """
        Args:
            clevr_dir (string): Root directory of CLEVR dataset
            train (bool): Tells if we are loading the train or the validation datasets
        """

        if split=='train':
            quest_json_filename = os.path.join(clevr_dir, 'CLEVR-Humans', 'CLEVR-Humans-train.json')
            self.img_dir = os.path.join(clevr_dir, 'images', 'train')
        else:
            quest_json_filename = os.path.join(clevr_dir, 'CLEVR-Humans', 'CLEVR-Humans-val.json')
            self.img_dir = os.path.join(clevr_dir, 'images', 'val')

        self.img_features_file = h5py.File(os.path.join(features_dir,'{}_resnet101_features.hdf5').format(split), 'r')['data']
        # self.img_features_file = h5py.File(os.path.join(features_dir,'{}.h5').format(split), 'r')['features'] # MAC networks original code

        cached_questions = quest_json_filename.replace('.json', '.pkl')
        if os.path.exists(cached_questions):
            print('==> using cached questions: {}'.format(cached_questions))
            with open(cached_questions, 'rb') as f:
                self.questions = pickle.load(f)
        else:
            with open(quest_json_filename, 'r') as json_file:
                self.questions = json.load(json_file)['questions']
            with open(cached_questions, 'wb') as f:
                pickle.dump(self.questions, f)
                
        self.clevr_dir = clevr_dir
        self.dictionaries = dictionaries
    
    def answer_weights(self):
        n = float(len(self.questions))
        answer_count = Counter(q['answer'].lower() for q in self.questions)
        weights = [n/answer_count[q['answer'].lower()] for q in self.questions]
        return weights
    
    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        current_question = self.questions[idx]
        img_filename = os.path.join(self.img_dir, current_question['image_filename'])
        # image = Image.open(img_filename).convert('RGB')
        image_id = int(img_filename.rsplit('_', 1)[1][:-4])

        image = self.img_features_file[image_id]

        _, H, W = image.shape
        position_encoding = get_positional_encoding(H, W)

        image = np.concatenate((image, position_encoding), axis=0)

        question = utils_clevr_humans.to_dictionary_indexes(self.dictionaries[0], current_question['question'])
        answer = utils_clevr_humans.to_dictionary_indexes(self.dictionaries[1], current_question['answer'])
        answer_class = self.dictionaries[2][answer.item()]

        answer = (answer-1) # convert to zero based indexing

        return image, question, len(question), answer, answer_class
