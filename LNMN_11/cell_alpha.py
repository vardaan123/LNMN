import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import collate_data, get_gpu_memory_map

class choose_a(nn.Module):
    def __init__(self):
        super(choose_a, self).__init__()

    def forward(self, x1, x2):
        return x1

class choose_b(nn.Module):
    def __init__(self):
        super(choose_b, self).__init__()

    def forward(self, x1, x2):
        return x2

class Node(nn.Module):
    def __init__(self, n_operations):
        super(Node, self).__init__()
        self.n_operations = n_operations # {min, max, +, .}
        self.op_weights = nn.Parameter(1e-3 * torch.randn(self.n_operations))
        self.choose_a = choose_a()
        self.choose_b = choose_b()

    def forward(self, x1, x2):
        # TODO: Use softmax instead of simple sum
        # print('op_weights requires_grad = {}'.format(self.op_weights.requires_grad))

        op_weights_norm = self.op_weights

        # try:
        o1 = op_weights_norm[0] * (torch.min(x1, x2))
        o2 = op_weights_norm[1] * (torch.max(x1, x2))
        o3 = op_weights_norm[2] * (torch.add(x1, x2))
        o4 = op_weights_norm[3] * (torch.mul(x1, x2))
        o5 = op_weights_norm[4] * (self.choose_a(x1, x2))
        o6 = op_weights_norm[5] * (self.choose_b(x1, x2))
        # except:
            # get_gpu_memory_map()
        out = (o1 + o2 + o3 + o4 + o5 + o6)

        out = F.normalize(out, p=2, dim=1)

        return out

        
class NetworkCell_4input(nn.Module):
    def __init__(self, n_operations, img_feat_dim, text_feat_dim, map_dim):
        super(NetworkCell_4input, self).__init__()
        self.n_operations = n_operations
        self.n_nodes = 3
        self.node_list = nn.ModuleList([Node(self.n_operations) for _ in range(self.n_nodes)])

        self.img_feat_dim = img_feat_dim
        self.text_feat_dim = text_feat_dim
        self.map_dim = map_dim

        # self.conv1 = nn.Conv2d(self.img_feat_dim, self.map_dim, 1) # 1x1 conv
        self.conv2 = nn.Conv2d(self.map_dim, 1, 1) # 1x1 conv

        # self.linear1 = nn.Linear(self.text_feat_dim, self.map_dim)
        self.reset_params()

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

    # @profile
    def forward(self, img_feat, attn_1, attn_2, c_txt):
        # self.train()
        '''
        Input
        img_feat : [batch, img_feat_dim, H, W]
        attn : [batch, map_dim, 1, 1]
        c_txt : [batch, D_txt]
        Output: [batch, map_dim, 1, 1]
        '''
        # print('img_feat = {}'.format(img_feat.size()))
        # print('attn = {}'.format(attn.size()))
        # print('c_txt = {}'.format(c_txt.size()))

        # img_feat_mapped = self.conv1(img_feat) # [batch, map_dim, H, W]
        # torch.cuda.synchronize()

        # c_txt_mapped = self.linear1(c_txt).unsqueeze(-1).unsqueeze(-1) # [batch, map_dim, 1, 1]
        # torch.cuda.synchronize()

        c_txt_mapped = c_txt.unsqueeze(-1).unsqueeze(-1) # [batch, map_dim, 1, 1]

        attn = self.node_list[0](attn_1, attn_2) # [batch, map_dim, H, W]

        out_node_1 = self.node_list[1](img_feat, attn) # [batch, map_dim, H, W]
        # torch.cuda.synchronize()

        out_node_2 = self.node_list[2](out_node_1, c_txt_mapped) # [batch, map_dim, H, W]
        # torch.cuda.synchronize()

        out = out_node_2 # [batch, map_dim, H, W]
        out_prev = out_node_1 # [batch, map_dim, H, W]

        out2 = self.conv2(out) # [batch, 1, H, W]

        return out2

class NetworkCell_3input(nn.Module):
    def __init__(self, n_operations, img_feat_dim, text_feat_dim, map_dim):
        super(NetworkCell_3input, self).__init__()
        self.n_operations = n_operations
        self.n_nodes = 2
        self.node_list = nn.ModuleList([Node(self.n_operations) for _ in range(self.n_nodes)])

        self.img_feat_dim = img_feat_dim
        self.text_feat_dim = text_feat_dim
        self.map_dim = map_dim

        # self.conv1 = nn.Conv2d(self.img_feat_dim, self.map_dim, 1) # 1x1 conv
        self.conv2 = nn.Conv2d(self.map_dim, 1, 1) # 1x1 conv

        # self.linear1 = nn.Linear(self.text_feat_dim, self.map_dim)
        self.reset_params()

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

    # @profile
    def forward(self, img_feat, attn, c_txt):
        # self.train()
        '''
        Input
        img_feat : [batch, img_feat_dim, H, W]
        attn : [batch, map_dim, 1, 1]
        c_txt : [batch, D_txt]
        Output: [batch, map_dim, 1, 1]
        '''
        # print('img_feat = {}'.format(img_feat.size()))
        # print('attn = {}'.format(attn.size()))
        # print('c_txt = {}'.format(c_txt.size()))

        # img_feat_mapped = self.conv1(img_feat) # [batch, map_dim, H, W]
        # torch.cuda.synchronize()

        # c_txt_mapped = self.linear1(c_txt).unsqueeze(-1).unsqueeze(-1) # [batch, map_dim, 1, 1]
        # torch.cuda.synchronize()

        c_txt_mapped = c_txt.unsqueeze(-1).unsqueeze(-1) # [batch, map_dim, 1, 1]

        out_node_0 = self.node_list[0](img_feat, attn) # [batch, map_dim, H, W]
        # torch.cuda.synchronize()

        out_node_1 = self.node_list[1](out_node_0, c_txt_mapped) # [batch, map_dim, H, W]
        # torch.cuda.synchronize()

        out = out_node_1 # [batch, map_dim, H, W]
        out_prev = out_node_0 # [batch, map_dim, H, W]

        out2 = self.conv2(out) # [batch, 1, H, W]

        return out2

class NetworkAnswerCell_4input(nn.Module):
    def __init__(self, n_operations, img_feat_dim, text_feat_dim, map_dim, mem_dim):
        super(NetworkAnswerCell_4input, self).__init__()
        self.n_operations = n_operations
        self.n_nodes = 3
        self.node_list = nn.ModuleList([Node(self.n_operations) for _ in range(self.n_nodes)])

        self.img_feat_dim = img_feat_dim
        self.text_feat_dim = text_feat_dim
        self.map_dim = map_dim
        self.mem_dim = mem_dim

        # self.conv1 = nn.Conv2d(self.img_feat_dim, self.map_dim, 1) # 1x1 conv
        self.linear3 = nn.Linear(2*self.map_dim + self.mem_dim, self.mem_dim)

        # self.linear1 = nn.Linear(self.text_feat_dim, self.map_dim)
        self.reset_params()

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

    # @profile
    def forward(self, img_feat, c_txt, attn_1, attn_2, mem_in):
        # self.train()
        '''
        Input
        img_feat : [batch, img_feat_dim, H, W]
        attn : [batch, map_dim, 1, 1]
        c_txt : [batch, D_txt]
        Output: [batch, map_dim, 1, 1]
        '''

        c_txt_mapped = c_txt.unsqueeze(-1).unsqueeze(-1) # [batch, map_dim, 1, 1]

        attn = self.node_list[0](attn_1, attn_2) # [batch, map_dim, H, W]

        out_node_1 = self.node_list[1](img_feat, attn) # [batch, map_dim, H, W]

        out_node_2 = self.node_list[2](out_node_1, c_txt_mapped) # [batch, map_dim, H, W]

        out = out_node_2 # [batch, map_dim, H, W]
        out_prev = out_node_1 # [batch, map_dim, H, W]

        out = out.sum(dim=-1).sum(dim=-1) # [batch, map_dim]
        
        out = self.linear3(torch.cat((c_txt, mem_in, out), dim=1)) # [batch, mem_dim]

        return out

class NetworkAnswerCell_3input(nn.Module):
    def __init__(self, n_operations, img_feat_dim, text_feat_dim, map_dim, mem_dim):
        super(NetworkAnswerCell_3input, self).__init__()
        self.n_operations = n_operations
        self.n_nodes = 2
        self.node_list = nn.ModuleList([Node(self.n_operations) for _ in range(self.n_nodes)])

        self.img_feat_dim = img_feat_dim
        self.text_feat_dim = text_feat_dim
        self.map_dim = map_dim
        self.mem_dim = mem_dim

        # self.conv1 = nn.Conv2d(self.img_feat_dim, self.map_dim, 1) # 1x1 conv
        self.linear3 = nn.Linear(2*self.map_dim + self.mem_dim, self.mem_dim)

        # self.linear1 = nn.Linear(self.text_feat_dim, self.map_dim)
        self.reset_params()

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

    # @profile
    def forward(self, img_feat, c_txt, attn, mem_in):
        # self.train()
        '''
        Input
        img_feat : [batch, img_feat_dim, H, W]
        attn : [batch, map_dim, 1, 1]
        c_txt : [batch, D_txt]
        Output: [batch, map_dim, 1, 1]
        '''

        c_txt_mapped = c_txt.unsqueeze(-1).unsqueeze(-1) # [batch, map_dim, 1, 1]

        out_node_0 = self.node_list[0](img_feat, attn) # [batch, map_dim, H, W]

        out_node_1 = self.node_list[1](out_node_0, c_txt_mapped) # [batch, map_dim, H, W]

        out = out_node_1 # [batch, map_dim, H, W]
        out_prev = out_node_0 # [batch, map_dim, H, W]

        out = out.sum(dim=-1).sum(dim=-1) # [batch, map_dim]
        
        out = self.linear3(torch.cat((c_txt, mem_in, out), dim=1)) # [batch, mem_dim]

        return out

class Answer(nn.Module): # or Describe
    def __init__(self, input_feat_dim, text_param_dim, map_dim, mem_dim):
        super(Answer, self).__init__()
        self.input_feat_dim = input_feat_dim
        self.text_param_dim = text_param_dim
        self.map_dim = map_dim
        self.mem_dim = mem_dim
        self.linear1 = nn.Linear(self.text_param_dim, self.map_dim)
        self.linear2 = nn.Linear(self.input_feat_dim, self.map_dim)
        self.linear3 = nn.Linear(self.map_dim + self.text_param_dim + self.mem_dim, self.mem_dim)
        self.n_attention_input = 1
        self.reset_params()

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

    def forward(self, img, text_param, attn, mem_in):
        '''
        Input
        img : [batch, input_feat_dim, H, W]
        text_param : [batch, D_txt]
        attn: [batch, 1, H, W]
        Output: [batch, 1, H, W]
        '''
        batch_size, _, H, W = img.size()
        text_param_transform = self.linear1(text_param) # [batch, map_dim]

        attn_softmax = F.softmax(attn.view(batch_size, -1), dim=1).view(batch_size, 1, H, W) # [batch, 1, H, W]

        attn_feat = torch.mul(img, attn_softmax).sum(dim=-1).sum(dim=-1) # [batch, input_feat_dim]

        attn_feat_mapped = self.linear2(attn_feat) # [batch, map_dim]

        element_wise_mult = torch.mul(text_param_transform, attn_feat_mapped) # [batch, map_dim]
        # element_wise_mult_norm = element_wise_mult.norm(p=2, dim=-1, keepdim=True) # [batch]
        # element_wise_mult = torch.div(element_wise_mult, element_wise_mult_norm) # [batch, map_dim]
        element_wise_mult = F.normalize(element_wise_mult, p=2, dim=1) # [batch, map_dim]

        out = self.linear3(torch.cat((text_param, mem_in, element_wise_mult), dim=1)) # [batch, mem_dim]
        return out

class Compare(nn.Module):
    def __init__(self, input_feat_dim, text_param_dim, map_dim, mem_dim):
        super(Compare, self).__init__()
        self.input_feat_dim = input_feat_dim
        self.text_param_dim = text_param_dim
        self.map_dim = map_dim
        self.mem_dim = mem_dim
        self.linear1 = nn.Linear(self.text_param_dim, self.map_dim)
        self.linear2 = nn.Linear(self.input_feat_dim, self.map_dim)
        self.linear3 = nn.Linear(self.input_feat_dim, self.map_dim)
        self.linear4 = nn.Linear(self.map_dim + self.text_param_dim + self.mem_dim, self.mem_dim)
        self.n_attention_input = 2

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

    def forward(self, img, text_param, attn1, attn2, mem_in):
        '''
        Input
        img : [batch, input_feat_dim, H, W]
        text_param : [batch, D_txt]
        attn1: [batch, 1, H, W]
        attn2: [batch, 1, H, W]
        Output: [batch, 1, H, W]
        '''
        batch_size, _, H, W = img.size()
        text_param_transform = self.linear1(text_param) # [batch, map_dim]

        attn1_softmax = F.softmax(attn1.view(batch_size, -1), dim=1).view(batch_size, 1, H, W) # [batch, 1, H, W]

        attn1_feat = torch.mul(img, attn1_softmax).sum(dim=-1).sum(dim=-1) # [batch, input_feat_dim]
        attn1_feat_mapped = self.linear2(attn1_feat) # [batch, map_dim]

        attn2_softmax = F.softmax(attn2.view(batch_size, -1), dim=1).view(batch_size, 1, H, W) # [batch, 1, H, W]

        attn2_feat = torch.mul(img, attn2_softmax).sum(dim=-1).sum(dim=-1) # [batch, input_feat_dim]
        attn2_feat_mapped = self.linear3(attn2_feat) # [batch, map_dim]

        element_wise_mult = (text_param_transform * attn1_feat_mapped * attn2_feat_mapped) # [batch, map_dim]
        # element_wise_mult_norm = element_wise_mult.norm(p=2, dim=-1, keepdim=True) # [batch]
        # element_wise_mult = torch.div(element_wise_mult, element_wise_mult_norm) # [batch, map_dim]
        element_wise_mult = F.normalize(element_wise_mult, p=2, dim=1) # [batch, map_dim]

        out = self.linear4(torch.cat((text_param, mem_in, element_wise_mult), dim=1)) # [batch, num_choices]
        return out

class NoOp(nn.Module):
    def __init__(self):
        super(NoOp, self).__init__()
        self.n_attention_input = 0
        
    def forward(self, img):
        return img