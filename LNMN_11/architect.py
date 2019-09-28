'''
Adapted from the repo: https://github.com/quark0/
'''

import torch
import numpy as np
import torch.nn as nn


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

class Architect(object):

    def __init__(self, model, device, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = 0.0 # TODO: change later if required
        self.model = model
        self.reg_coeff_op_loss = args.reg_coeff_op_loss
        # print('type(model) = {}'.format(type(self.model)))

        self.device = device
        self._criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.module.arch_parameters(),
                lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer, epoch_id):
        image, ques, q_len = input
        image, ques, q_len, target = image.to(self.device), ques.to(self.device), q_len, target.to(self.device)

        # loss = self._criterion(self.model(image, ques, q_len, epoch_id)[0], target)
        logits, _, op_loss = self.model(image, ques, q_len, epoch_id)
        loss = self._criterion(logits, target) -1 * self.reg_coeff_op_loss * op_loss.sum()

        theta = _concat(self.model.module.network_parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.module.network_parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)

        dtheta = _concat(torch.autograd.grad(loss, self.model.module.network_parameters())).data + self.network_weight_decay*theta
        # loss.backward()
        # dtheta = _concat(map(lambda x:x.grad, self.model.parameters())).data + self.network_weight_decay*theta

        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
        return unrolled_model

    # @profile
    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, epoch_id, unrolled):
        self.optimizer.zero_grad()
        # torch.cuda.synchronize()

        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer, epoch_id)
        else:
            self._backward_step(input_valid, target_valid, epoch_id)
            
        # torch.cuda.synchronize()
        self.optimizer.step()

    # @profile
    def _backward_step(self, input_valid, target_valid, epoch_id):
        image_valid, question_valid, q_len_valid = input_valid
        image_valid, question_valid, q_len_valid, target_valid = image_valid.to(self.device), question_valid.to(self.device), q_len_valid, target_valid.to(self.device)

        # loss = self._criterion(self.model(image_valid, question_valid, q_len_valid, epoch_id)[0], target_valid)
        logits, _, op_loss = self.model(image_valid, question_valid, q_len_valid, epoch_id)
        # print('op_loss = {}'.format(op_loss))
        # print(op_loss.requires_grad)
        loss = self._criterion(logits, target_valid) + 1 * self.reg_coeff_op_loss * op_loss.sum()

        loss.backward()

    # @profile
    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, epoch_id):
        unrolled_model = nn.DataParallel(self._compute_unrolled_model(input_train, target_train, eta, network_optimizer, epoch_id))
        # unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)

        image_valid, question_valid, q_len_valid = input_valid
        image_valid, question_valid, q_len_valid, target_valid = image_valid.to(self.device), question_valid.to(self.device), q_len_valid, target_valid.to(self.device)

        # unrolled_loss = self._criterion(unrolled_model(image_valid, question_valid, q_len_valid, epoch_id)[0], target_valid)
        logits, _, op_loss = unrolled_model(image_valid, question_valid, q_len_valid, epoch_id)
        unrolled_loss = self._criterion(logits, target_valid) + 1 * self.reg_coeff_op_loss * op_loss.sum()

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.module.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.module.network_parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train, epoch_id)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.module.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = g.new_tensor(g.data).requires_grad_(True)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.module.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            if 'op_weights' not in k:
                v_length = np.prod(v.size())
                params[k] = theta[offset: offset+v_length].view(v.size())
                offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict({k.replace('module.',''):v for k,v in model_dict.items()})
        return model_new.to(self.device)

    def _hessian_vector_product(self, vector, input, target, epoch_id, r=1e-2):
        R = r / _concat(vector).norm()

        for p, v in zip(self.model.module.network_parameters(), vector):
            # print('v = {}'.format(v))
            p.data.add_(R, v)
            # print('p = {}'.format(p.data))

        image, ques, q_len = input
        image, ques, q_len, target = image.to(self.device), ques.to(self.device), q_len, target.to(self.device)

        loss = self._criterion(self.model(image, ques, q_len, epoch_id)[0], target)
        grads_p = torch.autograd.grad(loss, self.model.module.arch_parameters())

        for p, v in zip(self.model.module.network_parameters(), vector):
            p.data.sub_(2*R, v)
        loss = self._criterion(self.model(image, ques, q_len, epoch_id)[0], target)
        grads_n = torch.autograd.grad(loss, self.model.module.arch_parameters())

        for p, v in zip(self.model.module.network_parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]
