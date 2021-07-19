# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Implementation adapted from Slimmable - https://github.com/JiahuiYu/slimmable_networks

import torch

class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification """
    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        return cross_entropy_loss.mean()



class KLLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification 
            output: output logits of the student network
            target: output logits of the teacher network
            T: temperature
            KL(p||q) = Ep \log p - \Ep log q
    """
    def forward(self, output, soft_logits, target=None, temperature=1., alpha=0.9):
        output, soft_logits = output / temperature, soft_logits / temperature
        soft_target_prob = torch.nn.functional.softmax(soft_logits, dim=1)
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        kd_loss = -torch.sum(soft_target_prob * output_log_prob, dim=1)
        if target is not None:
            n_class = output.size(1)
            target = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
            target = target.unsqueeze(1)
            output_log_prob = output_log_prob.unsqueeze(2)
            ce_loss = -torch.bmm(target, output_log_prob).squeeze()
            loss = alpha*temperature* temperature*kd_loss + (1.0-alpha)*ce_loss
        else:
            loss = kd_loss 
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CrossEntropyLossSmooth(torch.nn.modules.loss._Loss):
    def __init__(self, label_smoothing=0.1):
        super(CrossEntropyLossSmooth, self).__init__()
        self.eps = label_smoothing

    """ label smooth """
    def forward(self, output, target):
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        target = one_hot * (1 - self.eps) +  self.eps / n_class
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        loss = -torch.bmm(target, output_log_prob)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def f_divergence(q_logits, p_logits, alpha, iw_clip=1e3):
    assert isinstance(alpha, float)
    q_prob = torch.nn.functional.softmax(q_logits, dim=1).detach()
    p_prob = torch.nn.functional.softmax(p_logits, dim=1).detach()
    q_log_prob = torch.nn.functional.log_softmax(q_logits, dim=1) #gradient is only backpropagated here

    importance_ratio = p_prob / q_prob
    if abs(alpha) < 1e-3:
        importance_ratio = importance_ratio.clamp(0, iw_clip)
        f = -importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio.log() - 1.0
    elif abs(alpha - 1.0) < 1e-3:
        f = importance_ratio * importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio
    else:
        iw_alpha = torch.pow(importance_ratio, alpha)
        iw_alpha = iw_alpha.clamp(0, iw_clip)
        f = iw_alpha / alpha / (alpha - 1.0)
        f_base = 1.0 / alpha / (alpha - 1.0)
        rho_f = iw_alpha / alpha + f_base

    loss = torch.sum(q_prob * (f - f_base), dim=1) 
    grad_loss = -torch.sum(q_prob * rho_f * q_log_prob, dim=1)
    return loss, grad_loss


"""
It's often necessary to clip the maximum 
gradient value (e.g., 1.0) when using this adaptive KD loss
"""
class  AdaptiveLossSoft(torch.nn.modules.loss._Loss):
    def __init__(self, alpha_min=-1.0, alpha_max=1.0, iw_clip=5.0):
        super(AdaptiveLossSoft, self).__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.iw_clip = iw_clip

    def forward(self, output, target, alpha_min=None, alpha_max=None):
        alpha_min = alpha_min or self.alpha_min
        alpha_max = alpha_max or self.alpha_max
        
        loss_left, grad_loss_left = f_divergence(output, target, alpha_min, iw_clip=self.iw_clip)
        loss_right, grad_loss_right = f_divergence(output, target, alpha_max, iw_clip=self.iw_clip)

        ind = torch.gt(loss_left, loss_right).float()
        loss = ind * grad_loss_left + (1.0 - ind) * grad_loss_right

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


