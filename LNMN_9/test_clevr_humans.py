#!/u/username/anaconda3/envs/venv_0.4/bin/python -u
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

# from dataset import CLEVR, collate_data, transform
from dataset_clevr_humans import ClevrDataset

from model import Stack_NMN
from collections import Counter
import utils_clevr_humans
from utils import collate_data

use_counter = Counter()

def cv(input_t):
    return np.std(input_t)*1.0/np.mean(input_t)

def get_entropy_coeff(epoch_id, batch_id, n_batches):
    if epoch_id < 13:
        return 0.0
    elif epoch_id > 13:
        return 1.0
    return (batch_id * 1.0 / n_batches)


def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        try:
            os.makedirs(dst)
        except OSError:
            pass
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)


def val(model, criterion, optimizer, val_set, batch_size, device, summary_writer, n_epochs):
    dataset = iter(val_set)
    pbar = tqdm(dataset)

    model.train(False)
    family_correct = Counter()
    family_total = Counter()

    avg_loss = 0
    avg_acc = 0

    with torch.no_grad():
        for batch_id, (image, question, q_len, answer, family) in enumerate(pbar):
            image, question, answer = (
                image.to(device),
                question.to(device),
                answer.to(device),
            )

            # output, _, _ = model(image, question, q_len)
            output, _ = model(image, question, q_len)

            loss = criterion(output, answer)
            avg_loss += loss.item()

            correct = output.detach().argmax(1) == answer
            acc = torch.tensor(correct, dtype=torch.float32).sum() / batch_size

            avg_acc += acc

            for c, fam in zip(correct, family):
                if c:
                    family_correct[fam] += 1
                family_total[fam] += 1

    # for k, v in family_total.items():
    #     print('{}: {:.5f}\n'.format(k, family_correct[k] / v))
    avg_loss = avg_loss/(batch_id+1)
    avg_acc = avg_acc/(batch_id+1)

    print('Avg test. acc: {:.5f} loss: {:.5f}'.format(sum(family_correct.values()) / sum(family_total.values()), avg_loss))
    

# @profile
def main():
    parser = argparse.ArgumentParser(description='Stack-NMN')
    parser.add_argument('--embed_size', type=int, help='embedding dim. of question words', default=300)
    parser.add_argument('--lstm_hid_dim', type=int, help='hidden dim. of LSTM', default=256)
    parser.add_argument('--input_feat_dim', type=int, help='feat dim. of image features', default=1024)
    parser.add_argument('--map_dim', type=int, help='hidden dim. size of intermediate attention maps', default=512)
    parser.add_argument('--text_param_dim', type=int, help='hidden dim. of textual param.', default=512)
    parser.add_argument('--mlp_hid_dim', type=int, help='hidden dim. of mlp', default=512)
    parser.add_argument('--mem_dim', type=int, help='hidden dim. of mem.', default=512)
    parser.add_argument('--kb_dim', type=int, help='hidden dim. of conv features.', default=512)
    parser.add_argument('--max_stack_len', type=int, help='max. length of stack', default=8)
    parser.add_argument('--max_time_stamps', type=int, help='max. number of time-stamps for modules', default=9)
    parser.add_argument('--clevr_dir', type=str, help='Directory of CLEVR dataset', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--n_modules', type=int, default=9)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--features_dir', type=str, default='data')
    parser.add_argument('--clevr_feature_dir', type=str, default='/u/username/data/clevr_features/')
    parser.add_argument('--copy_data', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--reg_coeff', type=float, default=1e-1)
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--resume', action='store_true') # use only on slurm
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--use_half', action='store_true') # use only on slurm

    # SGDR hyper-params
    parser.add_argument('--T0', type=int, default=1)
    parser.add_argument('--Tmult', type=int, default=2)
    parser.add_argument('--eta_min', type=float, default=1e-5)

    args = parser.parse_args()
    print(args)
    '''
    with open('data/dic.pkl', 'rb') as f1:
        dic = pickle.load(f1)

    n_words = len(dic['word_dic']) + 1
    n_choices = len(dic['answer_dic'])

    print('n_words = {}, n_choices = {}'.format(n_words, n_choices))
    '''

    print('Building word dictionaries from all the words in the dataset...')

    dictionaries = utils_clevr_humans.build_dictionaries(args.clevr_dir)
    print('Building word dictionary completed!')

    print('Initializing CLEVR dataset...')

    # Build the model
    n_words = len(dictionaries[0])+1
    n_choices = len(dictionaries[1])

    print('n_words = {}, n_choices = {}'.format(n_words, n_choices))

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    writer = SummaryWriter(log_dir=args.model_dir)

    if args.copy_data:
        start_time = time.time()
        copytree(args.clevr_feature_dir, os.path.join(os.path.expandvars('$SLURM_TMPDIR'),'clevr_features/'))
        # copytree('/u/username/data/clevr_features/', '/Tmp/username/clevr_features/')
        # args.features_dir = '/Tmp/username/clevr_features/'
        args.features_dir = os.path.join(os.path.expandvars('$SLURM_TMPDIR'),'clevr_features/')
        print('data copy finished in {} sec.'.format(time.time() - start_time))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('device = {}'.format(device))

    model = Stack_NMN(args.max_stack_len, args.max_time_stamps, args.n_modules, n_choices, 3, n_words, args.embed_size, args.lstm_hid_dim, args.input_feat_dim, args.map_dim, args.mlp_hid_dim, args.mem_dim, args.kb_dim, args.kernel_size, False, device).to(device)
    # model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()

    # optimizer_1 = optim.Adam(map(lambda p:p[1], filter(lambda p:p[1].requires_grad and 'weight_mlp' not in p[0], model.named_parameters())), lr=args.lr, weight_decay=0e-3)

    # optimizer_2 = optim.Adam(map(lambda p:p[1], filter(lambda p:p[1].requires_grad and 'weight_mlp' in p[0], model.named_parameters())), lr=0e-8, weight_decay=0e-2)
    if args.optim == 'adam':
        optimizer = optim.Adam(map(lambda p:p[1], filter(lambda p:p[1].requires_grad, model.named_parameters())), lr=args.lr)
    elif args.optim == 'asgd':
        optimizer = optim.ASGD(map(lambda p:p[1], filter(lambda p:p[1].requires_grad, model.named_parameters())), lr=args.lr)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(map(lambda p:p[1], filter(lambda p:p[1].requires_grad, model.named_parameters())), lr=args.lr)
    elif args.optim == 'adamax':
        optimizer = optim.Adamax(map(lambda p:p[1], filter(lambda p:p[1].requires_grad, model.named_parameters())), lr=args.lr)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(map(lambda p:p[1], filter(lambda p:p[1].requires_grad, model.named_parameters())), lr=args.lr)
    elif args.optim == 'adadelta':
        optimizer = optim.Adadelta(map(lambda p:p[1], filter(lambda p:p[1].requires_grad, model.named_parameters())), lr=args.lr)
    elif args.optim == 'sgdr':
        optimizer = optim.SGD(map(lambda p:p[1], filter(lambda p:p[1].requires_grad, model.named_parameters())), lr=args.lr)

    clevr_val = ClevrDataset(args.clevr_dir, split='val', features_dir=args.features_dir, dictionaries=dictionaries)

    val_set = DataLoader(clevr_val, batch_size=args.batch_size, num_workers=4, collate_fn=collate_data)

    start_epoch = 0
    
    if len(args.ckpt)>0:
        model.load_state_dict({k:v for k,v in torch.load(args.ckpt).items() if 'embed' not in k}, strict=False)
        # start_epoch = int(args.ckpt.split('_')[-1].split('.')[0])
        # print('start_epoch = {}'.format(start_epoch))
        prev_embed = torch.load(args.ckpt)['embed.weight']
        model.embed.weight.data[:prev_embed.size(0), :].copy_(prev_embed)

    # print(model.embed.weight.data)
    
    if args.resume:
        model_ckpts = list(filter(lambda x:'ckpt_epoch' in x, os.listdir(args.model_dir)))
        
        if len(model_ckpts)>0:
            model_ckpts_epoch_ids = [int(filename.split('_')[-1].split('.')[0]) for filename in model_ckpts]
            start_epoch = max(model_ckpts_epoch_ids)
            latest_ckpt_file = os.path.join(args.model_dir, 'ckpt_epoch_{}.model'.format(start_epoch))
            model.load_state_dict(torch.load(latest_ckpt_file))
            print('Loaded ckpt file {}'.format(latest_ckpt_file))
            print('start_epoch = {}'.format(start_epoch))

    val(model, criterion, optimizer, val_set, args.batch_size, device, writer, args.n_epochs)

    writer.close()

if __name__ == '__main__':
    main()
