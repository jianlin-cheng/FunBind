import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    

class InfoNCELoss(nn.Module):
    def __init__(self, logit_scale=None, device="cuda"):
        super(InfoNCELoss, self).__init__()
         
        #self.logit_scale = torch.tensor(np.log(logit_scale), device=device)

        self.max_logit_scale = 100
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(logit_scale))
        self.device = device

    def forward(self, modality1_features, modality2_features):
        batch_size = modality1_features.shape[0]
        
        scaled_logit_scale = self.logit_scale.exp()
        
        
        logits = torch.matmul(modality1_features, modality2_features.T) * scaled_logit_scale

        logits_modality1 = logits
        logits_modality2 = logits.T

        labels = torch.arange(batch_size, device=self.device)

        total_loss = (
            F.cross_entropy(logits_modality1, labels) +
            F.cross_entropy(logits_modality2, labels)
        ) / 2

        return total_loss 
    
    def __repr__(self):
        return f"logit_scale={self.logit_scale}"
    

class DiversityLoss(nn.Module):
    def __init__(self):
        super(DiversityLoss, self).__init__()
        
        
    def forward(self, expert_outputs):
        
        num_experts = len(expert_outputs)
        expert_similarities = torch.zeros(num_experts, num_experts)
        
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                similarity = F.cosine_similarity(
                    expert_outputs[i].view(expert_outputs[i].size(0), -1),
                    expert_outputs[j].view(expert_outputs[j].size(0), -1)
                ).mean()
                expert_similarities[i, j] = similarity
                expert_similarities[j, i] = similarity
        
        
        return 1 + expert_similarities.mean()
    


class DiversityLoss2(nn.Module):
    def __init__(self, num_experts):
        super(DiversityLoss2, self).__init__()

        self.num_experts = num_experts
        
        
    def forward(self, expert_outputs):
        
        expert_outputs = torch.stack(expert_outputs, dim=1)
        normalized_outputs = F.normalize(expert_outputs, p=2, dim=-1)

        similarity_matrix = torch.matmul(normalized_outputs, normalized_outputs.transpose(1, 2))
        mask = torch.eye(self.num_experts, device=expert_outputs.device)
        masked_similarity = similarity_matrix * (1 - mask)

        diversity_loss = (masked_similarity.mean() + 1)/2

        return diversity_loss