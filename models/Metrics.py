import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from sklearn.metrics import auc, roc_curve, average_precision_score


class Similarity(nn.Module):
    def __init__(self, logit_scale):
        super(Similarity, self).__init__()

        max_temp = 100

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(logit_scale))
        self.register_buffer('max_logit_scale', torch.log(torch.tensor(max_temp)))
        
        
    def forward(self, modality1_features, modality2_features):
        scaled_logit_scale = self.logit_scale.exp()
        
        logits = torch.matmul(modality1_features, modality2_features.T) * scaled_logit_scale

        logits_modality1 = logits
        logits_modality2 = logits.T

        cosine_similarity1 = torch.diagonal(logits_modality1.softmax(dim=-1)).mean().item()
        cosine_similarity2 = torch.diagonal(logits_modality2.softmax(dim=-1)).mean().item()
        cosine_similarity = (cosine_similarity1 + cosine_similarity2)/2 

        return cosine_similarity
    
    def __repr__(self):
        return f"logit_scale={self.logit_scale}"




def compute_f(pr, rc):
    n = 2 * pr * rc
    d = pr + rc
    return torch.where(d != 0, n / d, torch.zeros_like(n))


def compute_s(ru, mi):
    return torch.sqrt(ru**2 + mi**2)



class PreRecF(nn.Module):
    def __init__(self, infor_accr=None, norm="cafa"):
        super(PreRecF, self).__init__()

        self.infor_accr = infor_accr

        valid_norms = {"cafa", "gt", "pred"}
        if norm not in valid_norms:
            raise ValueError(f"`norm` must be one of {valid_norms}, but got '{norm}'")
        self.norm = norm

    def compute_micro(self, labels, preds):
        true_positives = torch.sum((preds == 1) & (labels == 1)).item()
        false_positives = torch.sum((preds == 1) & (labels == 0)).item()
        false_negatives = torch.sum((preds == 0) & (labels == 1)).item()

        if true_positives + false_negatives != 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0.0

        if true_positives + false_positives != 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0.0

        if precision + recall != 0:
            fscore = 2 * precision * recall / (precision + recall)
        else:
            fscore = 0.0

        return {"Micro Precision": precision, "Micro Recall": recall, "Micro Fscore": fscore}


    def compute_macro(self, true_positives, false_positives, false_negatives, prc_normalize, rec_normalize, weighted=False):

        true_positives = torch.sum(true_positives, dim=1)
        false_positives = torch.sum(false_positives, dim=1)
        false_negatives = torch.sum(false_negatives, dim=1)
        
        num_predictions = true_positives + false_positives
        num_gt = true_positives + false_negatives

        precision = torch.where(
            num_predictions > 0,
            true_positives / num_predictions,
            torch.zeros_like(true_positives, dtype=torch.float)
        ).sum()

        recall = torch.where(
            num_gt > 0,
            true_positives / num_gt,
            torch.zeros_like(true_positives, dtype=torch.float)
        ).sum()

        prc = precision / prc_normalize if prc_normalize != 0 else torch.tensor(0.0, dtype=precision.dtype)
        rc = recall / rec_normalize if rec_normalize != 0 else torch.tensor(0.0, dtype=precision.dtype)

        mi = false_positives.sum()/rec_normalize if rec_normalize != 0 else torch.tensor(0.0, dtype=precision.dtype)
        ru = false_negatives.sum()/rec_normalize if rec_normalize != 0 else torch.tensor(0.0, dtype=precision.dtype)

        f = compute_f(prc, rc)
        s = compute_s(ru, mi)

        if weighted:
            return {"Weighted Precision": prc.item(), "Weighted Recall": rc.item(), 
                    "Weighted Fscore": f.item(), "Weighted Smin": s.item()}
        else:
            return {"Precision": prc.item(), "Recall": rc.item(), 
                    "Fscore": f.item(), "Smin": s.item()}
        
        
    def compute_scores(self, labels, preds):

        preds = preds > 0.5

        results = {}
        
        # results.update(self.compute_micro(labels, preds))

        gt_normalize = labels.shape[0]
        pred_normalize = (preds.sum(axis=1) > 0).sum()

        if self.norm == "cafa":
            norm1 = pred_normalize
            norm2 = gt_normalize
        elif self.norm == "pred":
            norm1 = pred_normalize
            norm2 = pred_normalize
        else:
            norm1 = gt_normalize
            norm2 = gt_normalize
            

        true_positives  = (preds == 1) & (labels == 1)
        false_positives = (preds == 1) & (labels == 0)
        false_negatives = (preds == 0) & (labels == 1)

        results.update(self.compute_macro(true_positives, false_positives, false_negatives, norm1, norm2))

        if self.infor_accr is not None:
            accr = torch.tensor(self.infor_accr, device=true_positives.device)

            true_positives = true_positives[:, :] * accr
            false_positives = false_positives[:, :] * accr
            false_negatives = false_negatives[:, :] * accr

            results.update(self.compute_macro(true_positives, false_positives, false_negatives, norm1, norm2, weighted=True))
        return results
    

class Retrive_at_k(nn.Module):
    def __init__(self, k=1):
        super(Retrive_at_k, self).__init__()
        self.k = k

    def forward(self, modality1_features, modality2_features, groundtruth_all_indices):

        size = modality1_features.size(0)

        similarity_scores = torch.matmul(modality1_features, modality2_features.T)
        top_k_indices = torch.topk(similarity_scores, k=self.k, dim=1).indices

        correct_matches = torch.zeros(size, dtype=torch.bool, device=modality1_features.device)
        groundtruth_tensors = [torch.tensor(gt, device=modality1_features.device) for gt in groundtruth_all_indices]

        for i in range(size):
            gt_indices = groundtruth_tensors[i]
            matches = torch.isin(top_k_indices[i], gt_indices)
            if torch.any(matches):
                correct_matches[i] = True

        success_at_k = correct_matches.float().sum() / size
        return success_at_k



class Retrieve_MRR(nn.Module):
    def __init__(self):
        super(Retrieve_MRR, self).__init__()

    def forward(self, modality1_features, modality2_features, groundtruth_all_indices):

        size = modality1_features.size(0)

        similarity_scores = torch.matmul(modality1_features, modality2_features.T)
        sorted_indices = torch.argsort(similarity_scores, dim=1, descending=True)

        reciprocal_ranks = torch.zeros(size, device=modality1_features.device)

        for i in range(size):
            gt_indices = torch.tensor(groundtruth_all_indices[i], device=modality1_features.device)

            ranks = (sorted_indices[i].unsqueeze(1) == gt_indices.unsqueeze(0)).nonzero(as_tuple=False)
            
            if ranks.numel() > 0:
                min_rank = ranks[:, 0].min().item() + 1
                reciprocal_ranks[i] = 1.0 / min_rank

        return reciprocal_ranks.mean()
    

    
'''

class Retrive_at_k(nn.Module):
    def __init__(self, k=1):
        super(Retrive_at_k, self).__init__()
        self.k = k
        
    def forward(self, modality1_features, modality2_features):
        
        size = modality1_features.size(0)

        similarity_scores = torch.matmul(modality1_features, modality2_features.T)
        top_k_indices = torch.topk(similarity_scores, k=self.k, dim=1).indices

        groundtruth = torch.arange(size, device=similarity_scores.device)
        correct_matches = (top_k_indices == groundtruth.unsqueeze(1))

        assert len(correct_matches) <= size

        true_positives = correct_matches.sum(dim=1)
        recall_at_k = true_positives.float().sum() / (size) # * self.k)

        return recall_at_k


class Retrieve_MRR(nn.Module):
    def __init__(self):
        super(Retrieve_MRR, self).__init__()
        
    def forward(self, modality1_features, modality2_features):
        
        size = modality1_features.size(0)
        groundtruth = torch.arange(size, device=modality1_features.device)

        similarity_scores = torch.matmul(modality1_features, modality2_features.T)
        sorted_indices = torch.argsort(similarity_scores, dim=1, descending=True)

        ranks = (sorted_indices == groundtruth.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1

        reciprocal_ranks = 1.0 / ranks.float()
        mrr = reciprocal_ranks.mean()

        return mrr
'''