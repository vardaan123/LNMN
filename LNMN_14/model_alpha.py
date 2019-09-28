import torch
import torch.nn as nn
import torch.nn.functional as F
from cell_alpha import *
import sys
import numpy as np

def entropy(probs):
    entropy_loss = -1 * torch.sum(F.softmax(probs, dim=1) * F.log_softmax(probs, dim=1))

    return entropy_loss

class Stack_NMN(nn.Module):
    def __init__(self, max_stack_len, max_time_stamps, n_modules, n_choices, n_nodes, vocab_size, embed_hidden, lstm_hid_dim, img_feat_dim, map_dim, mlp_hid_dim, mem_dim, kb_dim, kernel_size, module_id_grad, device):
        super(Stack_NMN, self).__init__()
        self.max_stack_len = max_stack_len # initially the stack pointer is in position -1.
        self.max_time_stamps = max_time_stamps
        self.lstm_hid_dim = lstm_hid_dim
        self.embed_hidden = embed_hidden
        self.vocab_size = vocab_size
        self.n_modules = 9
        self.img_feat_dim = img_feat_dim
        self.map_dim = map_dim
        # self.text_param_dim = text_param_dim
        self.kernel_size = kernel_size
        self.n_choices = n_choices
        self.n_nodes = n_nodes
        self.n_operations = 6

        self.mlp_hid_dim = mlp_hid_dim
        self.mem_dim = mem_dim
        self.kb_dim = kb_dim
        self.module_id_grad = module_id_grad
        
        self.embed = nn.Embedding(self.vocab_size, self.embed_hidden)
        self.lstm = nn.LSTM(self.embed_hidden, self.lstm_hid_dim, batch_first=True, bidirectional=True)

        self.device = device
        # controller parameters
        self.W_1 = nn.ModuleList(nn.Linear(2 * self.lstm_hid_dim, 2 * self.lstm_hid_dim, bias=False) for _ in range(self.max_time_stamps))
        self.b_1 = nn.Parameter(torch.zeros(2 * self.lstm_hid_dim, dtype=torch.float32))
        # self.b_1 = nn.Parameter(torch.randn(2 * self.lstm_hid_dim, dtype=torch.float32))
        self.W_2 = nn.Linear(4 * self.lstm_hid_dim, 2 * self.lstm_hid_dim)
        self.W_3 = nn.Linear(2*lstm_hid_dim, 1, bias=False)
        self.c_0 = nn.Parameter(torch.randn(2*self.lstm_hid_dim, dtype=torch.float32))

        self.weight_mlp = nn.Linear(2 * self.lstm_hid_dim, self.n_modules, bias=False)

        self.output_mlp = nn.Sequential(nn.Linear(2*self.lstm_hid_dim + self.mem_dim, self.mlp_hid_dim),\
            nn.ELU(),\
            nn.Linear(self.mlp_hid_dim, self.n_choices))

        self.img_conv = nn.Sequential(nn.Conv2d(self.img_feat_dim + 128, self.kb_dim, 1),\
            nn.ELU(),\
            nn.Conv2d(self.kb_dim, self.kb_dim, 1))

        # self.bn = nn.BatchNorm1d(2 * self.lstm_hid_dim, affine=False)
        # self.bn2 = nn.BatchNorm1d(self.n_modules, affine=False)

        # self.weight_mlp = nn.Linear(2 * self.lstm_hid_dim, self.n_modules, bias=False)
        # self.weight_mlp = nn.Linear(2 * self.lstm_hid_dim, self.n_modules)

        self.module_list = nn.ModuleList(NetworkCell_3input(self.n_operations, self.kb_dim, 2*self.lstm_hid_dim, self.map_dim) for _ in range(4))
        self.module_list.extend(nn.ModuleList(NetworkCell_4input(self.n_operations, self.kb_dim, 2*self.lstm_hid_dim, self.map_dim) for _ in range(2)))
        self.module_list.extend(nn.ModuleList(NetworkAnswerCell_3input(self.n_operations, self.kb_dim, 2*self.lstm_hid_dim, self.map_dim, self.mem_dim) for _ in range(1)))
        self.module_list.extend(nn.ModuleList(NetworkAnswerCell_4input(self.n_operations, self.kb_dim, 2*self.lstm_hid_dim, self.map_dim, self.mem_dim) for _ in range(1)))
        self.module_list.append(NoOp())

        # print('No. of modules = {}'.format(len(self.module_list)))

        self._criterion = nn.CrossEntropyLoss()

        self.img_feat_to_map_dim = nn.Conv2d(self.kb_dim, self.map_dim, 1) # 1x1 conv
        self.text_to_map_dim = nn.Linear(2*self.lstm_hid_dim, self.map_dim)

        self.reset_params()
        
        #************************************************************************
        # make changes to op_weights based on alpha
        for node in self.module_list[self.module_id_grad].node_list:
            node.op_weights.retain_grad()
        #************************************************************************
        
    def reset_params(self):
        for m_name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                # print('initializing {}'.format(m_name))
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.xavier_uniform_(m.weight.data)
                
                if m.bias is not None:
                    # nn.init.normal_(m.bias.data)
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.LSTM):
                # print('initializing {}'.format(m_name))
                for name, param in m.named_parameters():
                      if 'bias' in name:
                         nn.init.constant_(param, 0.0)
                      elif 'weight' in name:
                         nn.init.xavier_normal_(param)
            elif isinstance(m, nn.Conv2d):
                # print('initializing {}'.format(m_name))
                # nn.init.xavier_normal_(m.weight.data)
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.normal_(m.bias.data)
            else:
                # print('module {} not initialized'.format(m_name))
                pass

    def arch_parameters(self):
        arch_param = list(map(lambda x:x[1], filter(lambda x:'op_weights' in x[0], self.named_parameters())))
        return arch_param

    def network_parameters(self):
        net_param = map(lambda x:x[1], filter(lambda x:'op_weights' not in x[0], self.named_parameters()))

        return net_param

    def new(self):
        model_new = Stack_NMN(self.max_stack_len, self.max_time_stamps, self.n_modules, self.n_choices, self.n_nodes, self.vocab_size, self.embed_hidden, self.lstm_hid_dim, self.img_feat_dim, self.map_dim, self.mlp_hid_dim, self.mem_dim, self.kb_dim, self.kernel_size, self.device).to(self.device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _logits(self, image, question, q_len, target_ans):
        # print('flag 1')
        # self.zero_grad()
        # print(type(self))
        logits, _ = self(image, question, q_len)
        # print('flag 2')

        # loss = self._criterion(logits, target_ans)
        # loss.backward()

        # print('W_2 grad  = {}'.format(self.W_2.weight.grad))
        # print('conv2 grad  = {}'.format(self.module_list[0].conv2.weight.grad))
        # print('op_weights.grad  = {}'.format(self.module_list[0].node_list[0].op_weights.grad))

        return logits

    # @profile
    def forward(self, image, ques, q_len):
        '''
        image: [64, 512, 15, 10]
        ques
        '''
        # print(image)
        # TODO
        # sys.exit(0)
        image = self.img_conv(image)

        batch_size, _, H, W = image.size()
        total_ques_len = ques.size(1)
        # self.bn.train()
        # self.bn2.train()

        embed = self.embed(ques)

        embed_packed = nn.utils.rnn.pack_padded_sequence(embed, q_len, batch_first=True)
        # print('embed = {}'.format(embed.size()))

        self.lstm.flatten_parameters()

        packed_output, (h, _) = self.lstm(embed_packed) # h: [2, batch, lstm_hid_dim]
        # print('h = {}'.format(h.size()))

        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True) # [batch, n_max_words, 2*lstm_hid_dim]
        # print('lstm_out = {}'.format(lstm_out.size()))
        '''
        if torch.isnan(lstm_out).sum()>0:
            print('Nan in lstm_out')
        '''
        # torch.cuda.synchronize()

        h = h.transpose(0, 1).contiguous().view(image.size(0), -1) # [batch, 2*lstm_hid_dim]
        # print('h = {}'.format(h.size()))

        # c_prev = lstm_out.mean(dim=1) # initialize textual param. with hidden state of LSTM
        c_prev = self.c_0.repeat(batch_size, 1) # c_0 is a learnt parameter

        q = h

        feature_map_stack = torch.zeros(batch_size, H, W, self.max_stack_len, requires_grad=True).to(self.device) # Stack

        p = torch.zeros(batch_size, self.max_stack_len).to(self.device)
        # p[:, 0] = 1 # initialize stack pointer to bottom of stack
        p.index_fill_(1, torch.tensor([0]).to(self.device), 1)
        p.requires_grad_(True)

        imp_loss = 0
        '''
        max_layout = []
        for _ in range(batch_size):
            max_layout.append([])
        '''
        # ans_logits = torch.zeros((batch_size, self.n_choices), requires_grad=True).to(self.device) # [batch, n_choices]
        # print('ans_logits requires_grad = {}'.format(ans_logits.requires_grad))
        img_feat_mapped = self.img_feat_to_map_dim(image)
        mem_prev = torch.zeros((batch_size, self.mem_dim), requires_grad=True).to(self.device) # [batch, n_choices]

        for t in range(self.max_time_stamps):
            # print('t = {}'.format(t))
            u = self.W_2(torch.cat((self.W_1[t](q) + self.b_1, c_prev), dim=1)) # [batch, 2*lstm_hid_dim]

            # print('u = {}'.format(u))
            # u = torch.rand(u.size()).cuda()
            # print(self.W_1[t](q).size())
            # print(self.b_1.size())
            # print((self.W_1[t](q) + self.b_1).size())
            # u = self.bn(u)

            # print('mean = {}, std = {}'.format(u.mean(dim=0), u.std(dim=0)))
            # print('mean = {}, std = {}'.format(u.mean(dim=0).size(), u.std(dim=0).size()))

            w_t = self.weight_mlp(u) # [batch, n_modules]
            # torch.save(self.weight_mlp.weight, 'mlp_w.pt')
            # torch.save(self.weight_mlp.bias, 'mlp_b.pt')

            # w_t = self.bn2(w_t)

            # print('mean = {}, std = {}'.format(w_t.mean(dim=0), w_t.std(dim=0)))
            # print('mean = {}, std = {}'.format(w_t.mean(dim=0).size(), w_t.std(dim=0).size()))

            # print('w_t = {}'.format(w_t))
            imp_loss += entropy(w_t)

            w_t = F.softmax(w_t, dim=1) # [batch, n_modules]
            '''
            if torch.isnan(w_t).sum()>0:
                print('Nan in w_t')
            '''
            # imp_loss += (cv(w_t) ** 2).sum()
            
            # print(w_t.sum(dim=0))
            # print('max value at {}'.format(torch.max(w_t.sum(dim=0), dim=-1)[1]))
            '''
            tmp = torch.max(w_t, dim=-1)[1]
            for idx in range(batch_size):
                max_layout[idx].append(tmp[idx].item())
            '''
            element_wise_prod = torch.mul(u.unsqueeze(1), lstm_out)  # [batch, n_max_words, 2*lstm_hid_dim]
            batch_size, n_max_words, d = element_wise_prod.size()
            element_wise_prod = self.W_3(element_wise_prod.view(-1, d)).view(batch_size, n_max_words, -1).squeeze(-1) # [batch, n_max_words]
            # print('element_wise_prod = {}'.format(element_wise_prod.size()))

            element_wise_prod = F.softmax(element_wise_prod, dim=1) # [batch, n_max_words]

            # torch.cuda.synchronize()

            c = (element_wise_prod.unsqueeze(-1) * lstm_out).sum(dim=1) # [batch, embed_hidden]

            # torch.cuda.synchronize()

            c_prev = c.clone()
            '''
            if torch.isnan(c).sum()>0:
                print('Nan in c')
            '''
            text_feat_mapped = self.text_to_map_dim(c)
            # torch.cuda.synchronize()

            feature_map_stack_prev = feature_map_stack.clone()
            p_prev = p.clone()
            
            # stack_list = []
            # stack_p_list = []
            
            stack_list = torch.zeros(batch_size, self.n_modules, H, W, self.max_stack_len).to(self.device)
            stack_p_list = torch.zeros(batch_size, self.n_modules, self.max_stack_len).to(self.device)

            # mem = torch.zeros((batch_size, self.mem_dim), requires_grad=True).to(self.device) # [batch, n_choices]
            ans_out_list = []

            for module_id, module in enumerate(self.module_list):
                stack = feature_map_stack_prev # [batch, H, W, max_stack_len]
                stack_p = p_prev # [batch, max_stack_len]
                    
                if module_id < 4:
                    a = (stack * stack_p.unsqueeze(1).unsqueeze(1)).sum(dim=-1).unsqueeze(1) # [batch, 1, H, W] ; read from stack

                    # move ptr bw
                    '''
                    tmp_p = torch.zeros(batch_size, self.max_stack_len).to(self.device)
                    tmp_p.index_fill_(1, torch.tensor([0]).to(self.device), 1)
                    stack_p = F.conv1d(stack_p.unsqueeze(1), torch.tensor([0,0,1], dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0), padding=1).squeeze(1) + stack_p * tmp_p # decrement the stack pointer
                    '''
                    out = module(img_feat_mapped, a, text_feat_mapped) # [batch, 1, H, W]
                    out = out.permute(0, 2, 3, 1) # [batch, H, W, 1]

                    # move ptr fw
                    '''
                    tmp_p = torch.zeros(batch_size, self.max_stack_len).to(self.device)
                    tmp_p.index_fill_(1, torch.tensor([self.max_stack_len-1]).to(self.device), 1)
                    stack_p = F.conv1d(stack_p.unsqueeze(1), torch.tensor([1,0,0], dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0), padding=1).squeeze(1) + stack_p * tmp_p # increment the stack pointer
                    '''
                    stack = stack * (1 - stack_p.unsqueeze(1).unsqueeze(1)) + out * stack_p.unsqueeze(1).unsqueeze(1) # [batch, H, W, max_stack_len]

                elif module_id >= 4 and module_id <= 5:
                    
                    a1 = (stack * stack_p.unsqueeze(1).unsqueeze(1)).sum(dim=-1).unsqueeze(1) # [batch, 1, H, W] ; read from stack
                    
                    # move ptr bw                
                    tmp_p = torch.zeros(batch_size, self.max_stack_len).to(self.device)
                    tmp_p.index_fill_(1, torch.tensor([0]).to(self.device), 1)
                    stack_p = F.conv1d(stack_p.unsqueeze(1), torch.tensor([0,0,1], dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0), padding=1).squeeze(1) + stack_p * tmp_p # decrement the stack pointer

                    a2 = (stack * stack_p.unsqueeze(1).unsqueeze(1)).sum(dim=-1).unsqueeze(1) # [batch, 1, H, W] ; read from stack
                    
                    # move ptr bw
                    # stack_p = F.conv1d(stack_p.unsqueeze(1), torch.tensor([0,0,1], dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0), padding=1).squeeze(1) + stack_p * tmp_p # decrement the stack pointer

                    out = module(img_feat_mapped, a1, a2, text_feat_mapped) # [batch, 1, H, W]
                    out = out.permute(0, 2, 3, 1) # [batch, H, W, 1]

                    # move ptr fw
                    '''
                    tmp_p = torch.zeros(batch_size, self.max_stack_len).to(self.device)
                    tmp_p.index_fill_(1, torch.tensor([self.max_stack_len-1]).to(self.device), 1)
                    stack_p = F.conv1d(stack_p.unsqueeze(1), torch.tensor([1,0,0], dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0), padding=1).squeeze(1) + stack_p * tmp_p # increment the stack pointer
                    '''
                    stack = stack * (1 - stack_p.unsqueeze(1).unsqueeze(1)) + out * stack_p.unsqueeze(1).unsqueeze(1) # [batch, H, W, max_stack_len]

                elif module_id == 6:
                    # Answer or Compare module
                    
                    a = (stack * stack_p.unsqueeze(1).unsqueeze(1)).sum(dim=-1).unsqueeze(1) # [batch, 1, H, W] ; read from stack
                    # move ptr bw
                    
                    tmp_p = torch.zeros(batch_size, self.max_stack_len).to(self.device)
                    tmp_p.index_fill_(1, torch.tensor([0]).to(self.device), 1)
                    stack_p = F.conv1d(stack_p.unsqueeze(1), torch.tensor([0,0,1], dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0), padding=1).squeeze(1) + stack_p * tmp_p # decrement the stack pointer
                    
                    out = module(img_feat_mapped, text_feat_mapped, a, mem_prev) # [batch, n_choices]
                    # move ptr fw
                    '''
                    tmp_p = torch.zeros(batch_size, self.max_stack_len).to(self.device)
                    tmp_p.index_fill_(1, torch.tensor([self.max_stack_len-1]).to(self.device), 1)
                    stack_p = F.conv1d(stack_p.unsqueeze(1), torch.tensor([1,0,0], dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0), padding=1).squeeze(1) + stack_p * tmp_p # increment the stack pointer
                    '''
                    '''
                    attn_zero = torch.zeros(batch_size, H, W, 1).to(self.device)
                    stack = stack * (1 - stack_p.unsqueeze(1).unsqueeze(1)) + attn_zero * stack_p.unsqueeze(1).unsqueeze(1) # [batch, H, W, max_stack_len]
                    '''
                    # mem += (out * w_t[:, module_id].unsqueeze(-1)) # [batch, n_choices]
                    ans_out_list.append(out * w_t[:, module_id].unsqueeze(-1))

                elif module_id == 7:
                    # Answer or Compare module
                    
                    a1 = (stack * stack_p.unsqueeze(1).unsqueeze(1)).sum(dim=-1).unsqueeze(1) # [batch, 1, H, W] ; read from stack
                    # move ptr bw
                    tmp_p = torch.zeros(batch_size, self.max_stack_len).to(self.device)
                    tmp_p.index_fill_(1, torch.tensor([0]).to(self.device), 1)
                    stack_p = F.conv1d(stack_p.unsqueeze(1), torch.tensor([0,0,1], dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0), padding=1).squeeze(1) + stack_p * tmp_p # decrement the stack pointer

                    a2 = (stack * stack_p.unsqueeze(1).unsqueeze(1)).sum(dim=-1).unsqueeze(1) # [batch, 1, H, W] ; read from stack

                    # move ptr bw
                    
                    tmp_p = torch.zeros(batch_size, self.max_stack_len).to(self.device)
                    tmp_p.index_fill_(1, torch.tensor([0]).to(self.device), 1)
                    stack_p = F.conv1d(stack_p.unsqueeze(1), torch.tensor([0,0,1], dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0), padding=1).squeeze(1) + stack_p * tmp_p # decrement the stack pointer
                    
                    out = module(img_feat_mapped, text_feat_mapped, a1, a2, mem_prev) # [batch, n_choices]
                    
                    # move ptr fw
                    '''
                    tmp_p = torch.zeros(batch_size, self.max_stack_len).to(self.device)
                    tmp_p.index_fill_(1, torch.tensor([self.max_stack_len-1]).to(self.device), 1)
                    stack_p = F.conv1d(stack_p.unsqueeze(1), torch.tensor([1,0,0], dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0), padding=1).squeeze(1) + stack_p * tmp_p # increment the stack pointer
                    '''
                    '''
                    attn_zero = torch.zeros(batch_size, H, W, 1).to(self.device)
                    stack = stack * (1 - stack_p.unsqueeze(1).unsqueeze(1)) + attn_zero * stack_p.unsqueeze(1).unsqueeze(1) # [batch, H, W, max_stack_len]
                    '''
                    # mem += (out * w_t[:, module_id].unsqueeze(-1)) # [batch, n_choices]
                    ans_out_list.append(out * w_t[:, module_id].unsqueeze(-1))
                else:
                    pass

                # print('ans_logits = {}'.format(ans_logits))
                #******************************************************************************************
                # stack_list.append(stack)
                # stack_p_list.append(stack_p)
                stack_list[:,module_id,:,:,:] = stack.clone()
                stack_p_list[:,module_id,:] = stack_p.clone()

            # print('stack_list.requires_grad = {}'.format(stack_list.requires_grad))
            # print('stack_p_list.requires_grad = {}'.format(stack_p_list.requires_grad))
            #******************************************************************************************
            '''
            print('stack after')
            print('stack = {}'.format(stack))

            print('stack pointer after')
            print('stack pointer = {}'.format(stack_p))
            '''
            # stack_list = torch.stack(stack_list, dim=1) # [batch_size, n_modules, H, W, max_stack_len]
            # stack_p_list = torch.stack(stack_p_list, dim=1) # [batch_size, n_modules, max_stack_len]
            
            # print('stack_list = {}'.format(stack_list.size()))
            # print('stack_p_list = {}'.format(stack_p_list.size()))
            # print('w_t = {}'.format(w_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).size()))
            mem = torch.stack(ans_out_list).sum(dim=0)
            
            feature_map_stack = (stack_list * w_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
            temperature = 0.2
            p = F.softmax((stack_p_list * w_t.unsqueeze(-1)).sum(dim=1) / temperature, dim=1) # [batch_size, max_stack_len]

            mem_prev = mem
            '''
            if torch.isnan(p).sum()>0:
                print('Nan in p')

            if torch.isnan(feature_map_stack).sum()>0:
                print('Nan in feature_map_stack')
            '''
            # print(w_t)
            # print('p')
            # print(p)
            # print('ans_logits = {}'.format(ans_logits))
        # print(max_layout)
        
        # max_layout_flat = [y for x in max_layout for y in x]

        tmp_p = torch.zeros(batch_size, self.max_stack_len).to(self.device)
        # tmp_p[:, 0] = 1 # initialize stack pointer to bottom of stack
        tmp_p.index_fill_(1, torch.tensor([0]).to(self.device), 1)
        # tmp_p.requires_grad_(True)

        # print('before pop, stack = {}'.format(stack))
        a = (feature_map_stack * p.unsqueeze(1).unsqueeze(1)).sum(dim=-1).unsqueeze(1) # [batch, 1, H, W] ; read from stack
        # print('attn. map read = {}'.format(a))
        '''
        if torch.isnan(a).sum()>0:
            print('Nan in a')
        '''
        p = F.conv1d(p.unsqueeze(1), torch.tensor([0,0,1], dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0), padding=1).squeeze(1) + p * tmp_p # decrement the stack pointer
        '''
        if torch.isnan(p).sum()>0:
            print('Nan in p')
        '''
        ans_logits = self.output_mlp(torch.cat((q, mem), dim=1))

        '''
        if torch.isnan(ans_logits).sum()>0:
            print('Nan in ans_logits')
        '''
        imp_loss = imp_loss/self.max_time_stamps
        # return ans_logits, imp_loss, max_layout_flat
        # print('imp_loss = {}'.format(imp_loss))

        op_weights = [node.op_weights for node in self.module_list[self.module_id_grad].node_list]
        return ans_logits, imp_loss, op_weights
