from __future__ import print_function

import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    def __init__(self, device):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = 1.0

    def forward(self, text_features, image_features, t_label, i_targets):
        image_features = image_features / image_features.norm(dim = -1, keepdim = True)
        text_features = text_features / text_features.norm(dim = -1, keepdim = True)
        batch_size = text_features.shape[0] 
        batch_size_N = image_features.shape[0]
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
            i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float().to(self.device) 

        logits = torch.div(torch.matmul(text_features, image_features.T), self.temperature)
        #logits = logits.norm(dim = -1, keepdim = True)
        #print(f"logits = {logits}\n")

        #print(f"The shape of mask -- {mask.shape}\n{mask}")
        #print(f"The shape of logits -- {logits.shape}\n{logits}\n")

        # for numerical stability
        #logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        #logits = logits - logits_max.detach() 
        #exp_logits = torch.exp(logits)
        #print(f"exp_logits = {exp_logits}\n")
        #log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        #log_prob = torch.log(exp_logits)
        #mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) 
        #loss = - mean_log_prob_pos.mean()
        loss = torch.exp(logits)

        return loss


class ImgToProConLoss(nn.Module):
    def __init__(self, device):
        super(ImgToProConLoss, self).__init__()
        self.device = device
        self.temperature = 1.0

    def forward(self, image_features, text_prototypes, i_labels, p_labels):
        image_features = image_features / image_features.norm(dim = -1, keepdim = True)
        text_prototypes = text_prototypes / text_prototypes.norm(dim = -1, keepdim = True)
        batch_size = image_features.shape[0]
        prototypes_size = text_prototypes.shape[0]
        mask = torch.eq(i_labels.unsqueeze(1).expand(batch_size, prototypes_size), \
                        p_labels.unsqueeze(0).expand(batch_size, prototypes_size)).float().to(self.device)
        
        logits = torch.div(torch.matmul(image_features, text_prototypes.T), self.temperature)
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim = 1, keepdim = True))
        mean_log_prob_pos = (mask * log_prob).sum(dim = 1) / mask.sum(dim = 1)
        loss = - mean_log_prob_pos.mean()
        return loss