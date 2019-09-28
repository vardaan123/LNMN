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
# from model2 import Stack_NMN
from collections import Counter
import utils_clevr_humans
from utils_clevr_humans import collate_data, get_gpu_memory_map
from architect import Architect

use_counter = Counter()

def cv(input_t):
    return np.std(input_t)*1.0/np.mean(input_t)

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

def get_entropy_coeff(epoch_id):
    if epoch_id <= 19:
        return (1.0 - 2.0*epoch_id/49.0)
    else:
        return (11.0/49.0 - 6.0/49.0 * (epoch_id - 19.0))
    
# @profile
def train(epoch_id, model, architect, criterion, optimizer, train_set, val_set_architect, batch_size, device, summary_writer, n_epochs, lr, unrolled, reg_coeff):
    dataset = iter(train_set)
    pbar = tqdm(dataset)

    n_batches = len(dataset)

    model.train(True)

    avg_loss = 0
    avg_acc = 0

    for batch_id, (image, question, q_len, answer, _) in enumerate(pbar):
        image, question, answer = (
            image.to(device),
            question.to(device),
            answer.to(device),
        )
        # torch.cuda.synchronize()
        # print(torch.cuda.device_count())
        # torch.cuda.empty_cache()
        # get_gpu_memory_map()
        image_search, question_search, q_len_search, answer_search, _ = next(iter(val_set_architect))

        # torch.cuda.synchronize()
        # print('flag 1')
        # torch.cuda.empty_cache()
        # get_gpu_memory_map()
        architect.step((image, question, q_len), answer, (image_search, question_search, q_len_search), answer_search, lr, optimizer, epoch_id, unrolled=unrolled)
        # print('flag 2')
        # get_gpu_memory_map()

        # torch.cuda.synchronize()

        model.zero_grad()
        # output, imp_loss, max_layout_flat = model(image, question, q_len)
        # torch.cuda.empty_cache()
        # get_gpu_memory_map()
        output, imp_loss, _ = model(image, question, q_len, 0)
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()
        # get_gpu_memory_map()
        # use_counter.update(max_layout_flat)

        # print(use_counter)

        # print('CV = {}'.format(cv(np.asarray(list(map(lambda x: x[1], use_counter.most_common()))))))

        orig_loss = criterion(output, answer).sum()

        # print('imp_loss = {}, loss = {}'.format(imp_loss, orig_loss))
        # coeff = (1.0 - 2.0*epoch_id/(n_epochs-1))
        # coeff = get_entropy_coeff(epoch_id, batch_id, n_batches)
        coeff = get_entropy_coeff(epoch_id)

        # loss = orig_loss + coeff * reg_coeff * imp_loss.sum()
        loss = orig_loss -1 * coeff * reg_coeff * imp_loss.sum()
        
        if torch.isnan(loss).sum()>0:
            print('orig_loss = {}'.format(orig_loss))
            print('imp_loss = {}'.format(imp_loss))
            print('loss = {}'.format(loss))
            print('output = {}'.format(output))
            sys.exit(0)

        avg_loss += loss.item()

        # print('loss = {}'.format(loss.data[0]))
        
        loss.backward()
        '''
        #*********************************************
        print('W_2 grad  = {}'.format(model.W_2.weight.grad.sum()))
        print('conv2 grad  = {}'.format(model.module_list[0].conv2.weight.grad.sum()))
        # print('op_weights.grad  = {}'.format(model.module_list[0].node_list[0].op_weights.grad))
        #*********************************************
        '''
        # nn.utils.clip_grad_norm_(model.parameters(), 10)

        optimizer.step()
        # print('sum(norm) = {}'.format(sum(list(map(lambda x:x.norm(2), model.parameters())))))

        correct = output.detach().argmax(1) == answer
        acc = torch.tensor(correct, dtype=torch.float32).sum() / batch_size

        avg_acc += acc

        pbar.set_description(
            'Epoch: {}; Loss: {:.5f}; Acc: {:.5f}'.format(
                epoch_id + 1, loss.item(), acc
            )
        )

        summary_writer.add_scalar('loss/train', loss, batch_id)
        summary_writer.add_scalar('acc/train', acc, batch_id)


    avg_loss = avg_loss/(batch_id+1)
    avg_acc = avg_acc/(batch_id+1)

    summary_writer.add_scalar('loss_epoch/train', avg_loss, epoch_id)
    summary_writer.add_scalar('acc_epoch/train', avg_acc, epoch_id)

    print('Epoch {}, Avg train acc: {:.5f} loss: {:.5f}'.format(epoch_id, avg_acc, avg_loss))

    summary_writer.file_writer.flush()

def valid(epoch_id, model, criterion, optimizer, val_set, batch_size, device, summary_writer, n_epochs):
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
            output, _, _ = model(image, question, q_len, 0)

            loss = criterion(output, answer)
            avg_loss += loss.item()

            correct = output.detach().argmax(1) == answer
            acc = torch.tensor(correct, dtype=torch.float32).sum() / batch_size

            avg_acc += acc

            for c, fam in zip(correct, family):
                if c:
                    family_correct[fam] += 1
                family_total[fam] += 1

            summary_writer.add_scalar('loss/val', loss, batch_id)
            summary_writer.add_scalar('acc/val', acc, batch_id)
    # for k, v in family_total.items():
    #     print('{}: {:.5f}\n'.format(k, family_correct[k] / v))
    avg_loss = avg_loss/(batch_id+1)
    avg_acc = avg_acc/(batch_id+1)

    summary_writer.add_scalar('loss_epoch/val', avg_loss, epoch_id)
    summary_writer.add_scalar('acc_epoch/val', avg_acc, epoch_id)

    summary_writer.file_writer.flush()
    print('Epoch {}, Avg val. acc: {:.5f} loss: {:.5f}'.format(epoch_id, sum(family_correct.values()) / sum(family_total.values()), avg_loss))
    

# @profile
def main():
    parser = argparse.ArgumentParser(description='Stack-NMN')
    parser.add_argument('--embed_size', type=int, help='embedding dim. of question words', default=300)
    parser.add_argument('--lstm_hid_dim', type=int, help='hidden dim. of LSTM', default=256)
    parser.add_argument('--input_feat_dim', type=int, help='feat dim. of image features', default=1024)
    parser.add_argument('--map_dim', type=int, help='hidden dim. size of intermediate attention maps', default=512)
    # parser.add_argument('--text_param_dim', type=int, help='hidden dim. of textual param.', default=512)
    parser.add_argument('--mlp_hid_dim', type=int, help='hidden dim. of mlp', default=512)
    parser.add_argument('--mem_dim', type=int, help='hidden dim. of mem.', default=512)
    parser.add_argument('--kb_dim', type=int, help='hidden dim. of conv features.', default=512)
    parser.add_argument('--max_stack_len', type=int, help='max. length of stack', default=8)
    parser.add_argument('--max_time_stamps', type=int, help='max. number of time-stamps for modules', default=9)
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
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--resume', action='store_true') # use only on slurm
    parser.add_argument('--use_argmax', action='store_true')
    parser.add_argument('--reg_coeff_op_loss', type=float, default=1e-1)

    # DARTS args
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    args = parser.parse_args()

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

    # TODO: remove this later
    # args.features_dir = '/Tmp/username/clevr_features/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('device = {}'.format(device))

    model = Stack_NMN(args.max_stack_len, args.max_time_stamps, args.n_modules, n_choices, args.n_nodes, 0.0, n_words, args.embed_size, args.lstm_hid_dim, args.input_feat_dim, args.map_dim, args.mlp_hid_dim, args.mem_dim, args.kb_dim, args.kernel_size, device).to(device)
    # model = Stack_NMN(args.max_stack_len, args.max_time_stamps, args.n_modules, n_choices, n_words, args.embed_size, args.lstm_hid_dim, args.input_feat_dim, args.map_dim, args.text_param_dim, args.mlp_hid_dim, args.kernel_size, device).to(device)
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

    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()

    # optimizer_1 = optim.Adam(map(lambda p:p[1], filter(lambda p:p[1].requires_grad and 'weight_mlp' not in p[0], model.named_parameters())), lr=args.lr, weight_decay=0e-3)

    # optimizer_2 = optim.Adam(map(lambda p:p[1], filter(lambda p:p[1].requires_grad and 'weight_mlp' in p[0], model.named_parameters())), lr=0e-8, weight_decay=0e-2)
    optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.module.network_parameters()), lr=1e-4)
    # optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.network_parameters()), lr=args.lr)
    # optimizer = optim.Adam(map(lambda x:x[1], filter(lambda p:p[1].requires_grad, model.named_parameters())), lr=args.lr)

    architect = Architect(model, device, args)

    clevr_train = ClevrDataset(args.clevr_dir, split='train', features_dir=args.features_dir, dictionaries=dictionaries)
    clevr_val = ClevrDataset(args.clevr_dir, split='val', features_dir=args.features_dir, dictionaries=dictionaries)

    train_set = DataLoader(clevr_train, batch_size=args.batch_size, num_workers=0, collate_fn=collate_data)
    val_set = DataLoader(clevr_val, batch_size=args.batch_size, num_workers=0, collate_fn=collate_data)
    val_set_architect = DataLoader(clevr_val, batch_size=args.batch_size, num_workers=0, collate_fn=collate_data, sampler=torch.utils.data.RandomSampler(list(range(len(clevr_val)))))

    for epoch_id in range(start_epoch, args.n_epochs):
        train(epoch_id, model, architect, criterion, optimizer, train_set, val_set_architect, args.batch_size, device, writer, args.n_epochs, args.lr, args.unrolled, args.reg_coeff)
        valid(epoch_id, model, criterion, optimizer, val_set, args.batch_size, device, writer, args.n_epochs)

        with open('{}/ckpt_epoch_{}.model'.format(args.model_dir, str(epoch_id + 1)), 'wb') as f1:
            torch.save(model.module.state_dict(), f1)

    clevr_train.close()
    clevr_val.close()
    writer.close()

if __name__ == '__main__':
    main()
