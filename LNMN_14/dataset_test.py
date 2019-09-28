import os
import pickle

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import h5py
import utils_test
import torchvision
import json

from utils_test import get_positional_encoding

layout_map = {
u'_MoreNum': 7,\
u'_Count': 5,\
u'_SameProperty': 7,\
u'_Or': 3,\
u'_Transform':1,\
u'_FindSameProperty': 1,\
u'_Filter': 4,\
u'_Find': 0,\
u'_And': 3,\
u'_LessNum': 7,\
u'_Describe': 6,\
u'_Exist': 6,\
u'_EqualNum': 7,\
u'_Scene': 5,\
}

def get_ques_type(input_type):
    if input_type in ['count', 'exist']:
        return input_type
    elif input_type in ['less_than', 'greater_than', 'equal_integer']:
        return 'compare_number'
    elif input_type in ['query_color', 'query_shape', 'query_material', 'query_size']:
        return 'query_attribute'
    elif input_type in ['equal_material', 'equal_color', 'equal_size', 'equal_shape']:
        return 'compare_attribute'
    raise Exception('Invalid input type')

class ClevrDataset(Dataset):
    def __init__(self, clevr_dir, split, dictionaries, features_dir='data', layout_dir='', use_layout_supervision=False, use_half=False):
        """
        Args:
            clevr_dir (string): Root directory of CLEVR dataset
            train (bool): Tells if we are loading the train or the validation datasets
        """
        self.use_layout_supervision = use_layout_supervision

        if split=='train':
            if not use_half:
                quest_json_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_train_questions.json')
                self.img_dir = os.path.join(clevr_dir, 'images', 'train')
            else:
                quest_json_filename = os.path.join(clevr_dir, 'questions_half', 'CLEVR_train_questions.json')
                self.img_dir = os.path.join(clevr_dir, 'images', 'train')
        elif split =='val':
            quest_json_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_val_questions.json')
            self.img_dir = os.path.join(clevr_dir, 'images', 'val')
        else:
            quest_json_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_test_questions.json')
            self.img_dir = os.path.join(clevr_dir, 'images', 'test')

        self.img_features_file = h5py.File(os.path.join(features_dir,'{}_resnet101_features.hdf5').format(split), 'r')['data']
        # self.img_features_file = h5py.File(os.path.join(features_dir,'{}.h5').format(split), 'r')['features'] # MAC networks original code

        if self.use_layout_supervision:
            self.ques_with_layouts = json.load(open(os.path.join(layout_dir, 'CLEVR_{}_questions_gt_layout.json'.format(split))))

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
        self.max_layout_len = 19
    
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

        question = utils_test.to_dictionary_indexes(self.dictionaries[0], current_question['question'])
        # answer_class = self.dictionaries[2][answer.item()]
        return image, question, len(question)
