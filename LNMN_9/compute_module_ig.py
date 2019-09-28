import sys, os
import shutil
import pickle
from collections import Counter

import time
import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

from dataset import ClevrDataset

from model_alpha import Stack_NMN
from collections import Counter
import utils
from utils import collate_data, get_gpu_memory_map
from architect import Architect
from copy import deepcopy

def test(model, test_set, module_id_grad, device):
    dataset = iter(test_set)
    pbar = tqdm(dataset)

    model.train(False)
    # model.train(True)

    avg_acc = 0
    family_correct = Counter()
    family_total = Counter()

    op_weights_orig = [node.op_weights for node in model.module_list[module_id_grad].node_list]
    n_points = 10

    attr_global = torch.zeros(6).to(device)

    for batch_id, (image, question, q_len, answer, ques_type) in enumerate(pbar):
        image, question, answer = (
            image.to(device),
            question.to(device),
            answer.to(device),
        )

        attr = torch.zeros(6).to(device)

        for alpha in np.linspace(0.0, 1.0, n_points+1)[1:]:
            alpha = alpha.item()
            
            model_alpha = deepcopy(model)
            model_alpha.train(True)
            model_alpha.zero_grad()

            for node in model_alpha.module_list[module_id_grad].node_list:
                tmp = node.op_weights
                node.op_weights.data.copy_((1-alpha)*torch.zeros_like(tmp) + alpha * tmp)

            output, _, op_weight_list = model_alpha(image, question, q_len) # output: [batch, n_choices]

            # print(output.requires_grad)

            correct = output.detach().argmax(1) == answer
            output_probs = torch.gather(output, dim=1, index=output.argmax(1).unsqueeze(-1))

            # print(output_probs.requires_grad)

            output_probs.backward(torch.ones_like(output_probs))

            for i, op_weight in enumerate(op_weight_list):
                # print('op_weights_orig = {}'.format(op_weights_orig[i]))
                # print('op_weight.grad = {}'.format(op_weight.grad))
                # print('prod = {}'.format(op_weights_orig[i] * op_weight.grad))
                attr += (op_weights_orig[i] * op_weight.grad)

            # get_gpu_memory_map()

        attr = torch.div(attr, (n_points)*1.0)
        print(attr.data)
        # print('attr = {}'.format(attr))

        attr_global += attr.abs()
        # print('attr_global = {}'.format(attr_global))
        # get_gpu_memory_map()
        '''
        for c, fam in zip(correct, ques_type):
            if c:
                family_correct[fam] += 1
            family_total[fam] += 1
        '''

    print('module_id_grad = {}, attr_global = {}'.format(module_id_grad, attr_global.data))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Stack-NMN')
    parser.add_argument('--embed_size', type=int, help='embedding dim. of question words', default=300)
    parser.add_argument('--lstm_hid_dim', type=int, help='hidden dim. of LSTM', default=256)
    parser.add_argument('--input_feat_dim', type=int, help='feat dim. of image features', default=1024)
    parser.add_argument('--map_dim', type=int, help='hidden dim. size of intermediate attention maps', default=128)
    # parser.add_argument('--text_param_dim', type=int, help='hidden dim. of textual param.', default=512)
    parser.add_argument('--mlp_hid_dim', type=int, help='hidden dim. of mlp', default=512)
    parser.add_argument('--mem_dim', type=int, help='hidden dim. of mem.', default=512)
    parser.add_argument('--kb_dim', type=int, help='hidden dim. of conv features.', default=512)
    parser.add_argument('--max_stack_len', type=int, help='max. length of stack', default=5)
    parser.add_argument('--max_time_stamps', type=int, help='max. number of time-stamps for modules', default=5)
    parser.add_argument('--clevr_dir', type=str, help='Directory of CLEVR dataset', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--n_modules', type=int, default=7) # includes 1 NoOp module
    parser.add_argument('--n_nodes', type=int, default=2) # TODO: change/tune later
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--features_dir', type=str, default='data')
    parser.add_argument('--clevr_feature_dir', type=str, default='/u/username/data/clevr_features/')
    parser.add_argument('--copy_data', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--reg_coeff', type=float, default=1e-2)
    parser.add_argument('--ckpt', type=str, required=True)
    # parser.add_argument('--use_argmax', action='store_true')
    parser.add_argument('--module_id', type=int, default=0)

    # DARTS args
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    args = parser.parse_args()

    print('Building word dictionaries from all the words in the dataset...')

    dictionaries = utils.build_dictionaries(args.clevr_dir)
    print('Building word dictionary completed!')

    print('Initializing CLEVR dataset...')

    # Build the model
    n_words = len(dictionaries[0])+1
    n_choices = len(dictionaries[1])

    print('n_words = {}, n_choices = {}'.format(n_words, n_choices))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Stack_NMN(args.max_stack_len, args.max_time_stamps, args.n_modules, n_choices, args.n_nodes, n_words, args.embed_size, args.lstm_hid_dim, args.input_feat_dim, args.map_dim, args.mlp_hid_dim, args.mem_dim, args.kb_dim, args.kernel_size, args.module_id, device).to(device)
    
    # load checkpoint
    model.load_state_dict(torch.load(args.ckpt))

    clevr_test = ClevrDataset(args.clevr_dir, split='val', features_dir=args.features_dir, dictionaries=dictionaries)
    test_set = DataLoader(clevr_test, batch_size=args.batch_size, num_workers=1, collate_fn=collate_data)

    test(model, test_set, args.module_id, device)


